#!/usr/bin/env python
"""B5 via the OFFICIAL ASFormer code (third_party/ASFormer, pinned, 1-line NumPy patch in eval.py).

Reuses the MS-TCN data layout exported by scripts/run_mstcn_official.py (data/mstcn/ev9v_<hash>), runs the authors'
main.py --action=train/predict with their constants (120 epochs, lr 5e-4, 10 layers, 64 maps, bz 1, channel mask 0.3),
computes their eval.py metrics per bank, and exports per-stream softmax through their own Trainer.model.
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
from scripts.run_mstcn_official import official_metrics_per_bank  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.compose.banks import multi_bank, native_bank, pair_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402

PINNED = "e1bbe4f3ed083748f91467c51a63ac2a8b9277ad"
NUM_EPOCHS = 120  # hard-coded in the official main.py; predict uses epoch-120.model
FEATURES_DIM = 2048


def main() -> None:
    ap = base_parser("Run official ASFormer on the shared recipe banks (data layout from run_mstcn_official.py)")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); ont = load_ontology()
    repo = paths.repo_root / "third_party" / "ASFormer"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"ASFormer commit {head} != pinned {PINNED}")
    classes = list(ont.classes(cfg["task"]))
    ds = f"ev9v_{args.ckpt_hash}"
    data_root = paths.data_root / "mstcn" / ds
    if not (data_root / "splits" / "train.split1.bundle").is_file():
        raise SystemExit(f"export the MS-TCN layout first: scripts/run_mstcn_official.py --skip-train ({data_root})")
    link = repo / "data"
    if not link.exists():
        link.symlink_to((paths.data_root / "mstcn").resolve(), target_is_directory=True)
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    banks = {"train": multi_bank(idx, "train", cfg)}
    for split in ("validation", "test"):
        banks[f"pairs_{split}"] = pair_bank(idx, split, cfg); banks[f"multi_{split}"] = multi_bank(idx, split, cfg); banks[f"native_{split}"] = native_bank(idx, split)
    eval_ids = [r.recipe_id for k, b in banks.items() if k != "train" for r in b.recipes]
    run_dir = new_run_dir(cfg, "b5_asformer_official")
    model_dir = f"models_{ds}"; result_dir = f"results_{ds}"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    common = [f"--dataset={ds}", "--split=1", f"--model_dir={model_dir}", f"--result_dir={result_dir}"]
    print(json.dumps({"dataset": ds, "eval_streams": len(eval_ids), "commit": head}), flush=True)
    if not args.skip_train:
        subprocess.run([sys.executable, "main.py", "--action=train", *common], cwd=repo, env=env, check=True)
    shutil.rmtree(repo / result_dir / ds, ignore_errors=True)
    subprocess.run([sys.executable, "main.py", "--action=predict", *common], cwd=repo, env=env, check=True)
    # their eval.py reads ./results/<ds>/split_1 — point a results dir at ours, then evaluate per bank with their functions
    per_bank = official_metrics_per_bank_asformer(repo, ds, result_dir, banks)
    (run_dir / "official_eval_per_bank.json").write_text(json.dumps(per_bank, indent=1)); print(json.dumps(per_bank, indent=1), flush=True)

    sys.path.insert(0, str(repo))
    import torch  # noqa: E402
    from model import Trainer  # noqa: E402  (official)
    trainer = Trainer(10, 2, 2, 64, FEATURES_DIM, len(classes), 0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device); trainer.model.load_state_dict(torch.load(repo / model_dir / ds / "split_1" / f"epoch-{NUM_EPOCHS}.model", map_location=device)); trainer.model.eval()
    pred_dir = paths.cache_root / "asformer_official" / args.ckpt_hash; pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sid in eval_ids:
            x = torch.from_numpy(np.load(data_root / "features" / f"{sid}.npy")).unsqueeze(0).to(device)
            prob = torch.softmax(trainer.model(x, torch.ones(x.size(), device=device))[-1], dim=1)[0].T.cpu().numpy()
            np.savez_compressed(pred_dir / f"{sid}.npz", prob=prob.astype(np.float32))
    shutil.copy(repo / model_dir / ds / "split_1" / f"epoch-{NUM_EPOCHS}.model", paths.checkpoints_root / f"{run_dir.name}_epoch{NUM_EPOCHS}.model")
    (run_dir / "pred_dir.txt").write_text(str(pred_dir)); print(f"pred_dir={pred_dir}\nrun_dir={run_dir}")


def official_metrics_per_bank_asformer(repo: Path, ds: str, result_dir: str, banks: dict) -> dict:
    """Same computation as the MS-TCN per-bank metrics but reading ASFormer's results dir layout."""
    import importlib
    sys.path.insert(0, str(repo)); ev = importlib.import_module("eval")
    out = {}
    for name, b in banks.items():
        if name == "train":
            continue
        overlap = [0.1, 0.25, 0.5]; tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3); correct = total = 0; edit = 0.0; n = 0
        for r in b.recipes:
            rec_path = repo / result_dir / ds / "split_1" / r.recipe_id
            if not rec_path.is_file():
                continue
            gt = (repo / "data" / ds / "groundTruth" / f"{r.recipe_id}.txt").read_text().split("\n")[:-1]
            rec = rec_path.read_text().split("\n")[1].split(); m = min(len(gt), len(rec)); gt, rec = gt[:m], rec[:m]
            correct += sum(1 for a, p in zip(gt, rec) if a == p); total += m; edit += ev.edit_score(rec, gt); n += 1
            for s_, o in enumerate(overlap):
                a, b_, c = ev.f_score(rec, gt, o); tp[s_] += a; fp[s_] += b_; fn[s_] += c
        f1 = []
        for s_ in range(3):
            p_ = tp[s_] / max(tp[s_] + fp[s_], 1e-9); r_ = tp[s_] / max(tp[s_] + fn[s_], 1e-9); f1.append(float(np.nan_to_num(2 * p_ * r_ / max(p_ + r_, 1e-9)) * 100))
        out[name] = {"n_streams": n, "frame_acc": 100.0 * correct / max(total, 1), "edit": edit / max(n, 1), "f1@10": f1[0], "f1@25": f1[1], "f1@50": f1[2]}
    return out


if __name__ == "__main__":
    main()
