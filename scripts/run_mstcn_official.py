#!/usr/bin/env python
"""B4 via the OFFICIAL MS-TCN code (third_party/ms-tcn, pinned, Python-3 patch only).

Steps: (1) build the shared recipe banks; (2) export them in the MS-TCN data layout (features zero-padded to the
code's hard-coded 2048-d, groundTruth label files, mapping, split bundles); (3) run `main.py --action=train` and
`--action=predict` with the authors' constants; (4) run their `eval.py`; (5) load the epoch-50 model through their
own `model.MultiStageModel` to export per-stream softmax for our routing metrics.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.compose.banks import Bank, multi_bank, native_bank, pair_bank, render_bank, save_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402

FEATURES_DIM = 2048   # hard-coded in the official main.py; our 512-d features are zero-padded
NUM_EPOCHS = 50       # hard-coded in the official main.py; predict uses epoch-50.model
PINNED = "33ed91c0c7576650a2367efc602553af4c5295b1"


def export_layout(root: Path, classes: list[str], train: list, evals: list, streams: dict) -> None:
    for d in ("features", "groundTruth", "splits"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "mapping.txt").write_text("".join(f"{i} {c}\n" for i, c in enumerate(classes)))
    for sid, s in streams.items():
        feat = np.zeros((FEATURES_DIM, s.feat.shape[0]), dtype=np.float32); feat[: s.feat.shape[1]] = s.feat.T
        np.save(root / "features" / f"{sid}.npy", feat)
        (root / "groundTruth" / f"{sid}.txt").write_text("".join(classes[int(l)] + "\n" for l in s.labels))
    (root / "splits" / "train.split1.bundle").write_text("".join(f"{sid}.txt\n" for sid in train))
    (root / "splits" / "test.split1.bundle").write_text("".join(f"{sid}.txt\n" for sid in evals))


def official_metrics_per_bank(repo: Path, ds: str, banks: dict, classes: list[str]) -> dict:
    """Frame accuracy, edit score and F1@{10,25,50} per bank, computed with the authors' own eval.py functions."""
    sys.path.insert(0, str(repo))
    import importlib
    ev = importlib.import_module("eval")  # official ms-tcn eval.py (patched for py3 only)
    out = {}
    for name, b in banks.items():
        if name == "train":
            continue
        overlap = [0.1, 0.25, 0.5]; tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3); correct = total = 0; edit = 0.0; n = 0
        for r in b.recipes:
            gt = (repo / "data" / ds / "groundTruth" / f"{r.recipe_id}.txt").read_text().split("\n")[:-1]
            rec_path = repo / "results" / ds / "split_1" / r.recipe_id
            if not rec_path.is_file():
                continue
            rec = rec_path.read_text().split("\n")[1].split()
            m = min(len(gt), len(rec)); gt, rec = gt[:m], rec[:m]
            correct += sum(1 for a, p in zip(gt, rec) if a == p); total += m
            edit += ev.edit_score(rec, gt); n += 1
            for s_, o in enumerate(overlap):
                tp1, fp1, fn1 = ev.f_score(rec, gt, o); tp[s_] += tp1; fp[s_] += fp1; fn[s_] += fn1
        f1 = []
        for s_ in range(3):
            prec = tp[s_] / max(tp[s_] + fp[s_], 1e-9); rec_ = tp[s_] / max(tp[s_] + fn[s_], 1e-9)
            f1.append(float(np.nan_to_num(2 * prec * rec_ / max(prec + rec_, 1e-9)) * 100))
        out[name] = {"n_streams": n, "frame_acc": 100.0 * correct / max(total, 1), "edit": edit / max(n, 1),
                     "f1@10": f1[0], "f1@25": f1[1], "f1@50": f1[2]}
    return out


def main() -> None:
    ap = base_parser("Run official MS-TCN on the shared recipe banks")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); ont = load_ontology()
    repo = paths.repo_root / "third_party" / "ms-tcn"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"ms-tcn commit {head} != pinned {PINNED}")
    classes = list(ont.classes(cfg["task"]))
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    cache = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    banks: dict[str, Bank] = {"train": multi_bank(idx, "train", cfg)}
    for split in ("validation", "test"):
        banks[f"pairs_{split}"] = pair_bank(idx, split, cfg); banks[f"multi_{split}"] = multi_bank(idx, split, cfg)
        banks[f"native_{split}"] = native_bank(idx, split)
    rec_dir = paths.data_root / "manifests" / "recipes"
    for b in banks.values():
        save_bank(b, rec_dir, args.ckpt_hash)
    streams = {}
    for name, b in banks.items():
        for s in render_bank(b, loader):
            streams[s.recipe.recipe_id] = s
    train_ids = [r.recipe_id for r in banks["train"].recipes]
    eval_ids = [r.recipe_id for k, b in banks.items() if k != "train" for r in b.recipes]
    ds = f"ev9v_{args.ckpt_hash}"
    data_root = paths.data_root / "mstcn" / ds
    export_layout(data_root, classes, train_ids, eval_ids, streams)
    link = repo / "data"
    if not link.exists():
        link.symlink_to((paths.data_root / "mstcn").resolve(), target_is_directory=True)
    run_dir = new_run_dir(cfg, "b4_mstcn_official")
    (run_dir / "banks.json").write_text(json.dumps({k: len(b.recipes) for k, b in banks.items()}, indent=1))
    print(json.dumps({"dataset": ds, "train_streams": len(train_ids), "eval_streams": len(eval_ids), "commit": head}), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if not args.skip_train:
        subprocess.run([sys.executable, "main.py", "--action=train", f"--dataset={ds}", "--split=1"], cwd=repo, env=env, check=True)
    shutil.rmtree(repo / "results" / ds, ignore_errors=True)  # stale predictions from earlier exports
    subprocess.run([sys.executable, "main.py", "--action=predict", f"--dataset={ds}", "--split=1"], cwd=repo, env=env, check=True)
    ev = subprocess.run([sys.executable, "eval.py", f"--dataset={ds}", "--split=1"], cwd=repo, env=env, check=True, capture_output=True, text=True)
    (run_dir / "official_eval_all.txt").write_text(ev.stdout); print("official eval.py on all eval streams:\n" + ev.stdout, flush=True)
    per_bank = official_metrics_per_bank(repo, ds, banks, classes)
    (run_dir / "official_eval_per_bank.json").write_text(json.dumps(per_bank, indent=1)); print(json.dumps(per_bank, indent=1), flush=True)

    # per-stream softmax through the authors' model class (no code change), for routing metrics
    sys.path.insert(0, str(repo))
    import torch  # noqa: E402
    from model import MultiStageModel  # noqa: E402  (official)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStageModel(4, 10, 64, FEATURES_DIM, len(classes)).to(device)
    model.load_state_dict(torch.load(repo / "models" / ds / "split_1" / f"epoch-{NUM_EPOCHS}.model", map_location=device))
    model.eval()
    pred_dir = paths.cache_root / "mstcn_official" / args.ckpt_hash; pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sid in eval_ids:
            x = torch.from_numpy(np.load(data_root / "features" / f"{sid}.npy")).unsqueeze(0).to(device)
            mask = torch.ones(1, len(classes), x.shape[2], device=device)
            prob = torch.softmax(model(x, mask)[-1], dim=1)[0].T.cpu().numpy()
            np.savez_compressed(pred_dir / f"{sid}.npz", prob=prob.astype(np.float32))
    shutil.copy(repo / "models" / ds / "split_1" / f"epoch-{NUM_EPOCHS}.model", paths.checkpoints_root / f"{run_dir.name}_epoch{NUM_EPOCHS}.model")
    (run_dir / "pred_dir.txt").write_text(str(pred_dir))
    print(f"pred_dir={pred_dir}\nrun_dir={run_dir}")


if __name__ == "__main__":
    main()
