#!/usr/bin/env python
"""Evaluate EchoPrime's released view classifier (ConvNeXt-Base, 11 coarse views, first frame) on the EV9V test split.

Externally-pretrained track: no training, no fine-tuning. Uses the authors' preprocessing (utils.crop_and_scale, their
mean/std, first frame) and their weights (model_data/weights/view_classifier.pt). EV9V codes are mapped to the coarse
families both models share; EchoPrime classes with no EV9V counterpart (A2C, A3C, SSN, Doppler_*) count as errors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.metrics_classification import classification_summary  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402

PINNED = "03874a5f203695f38968068c21584656d475b6b1"
# EchoPrime COARSE_VIEWS -> EV9V family (None = unsupported on EV9V)
TO_FAMILY = {"A2C": None, "A3C": None, "A4C": "A4C", "A5C": "A5C", "Apical_Doppler": None, "Doppler_Parasternal_Long": None,
             "Doppler_Parasternal_Short": None, "Parasternal_Long": "PLAX", "Parasternal_Short": "PSAX", "SSN": None, "Subcostal": "SC4C"}
EV9V_TO_FAMILY = {"PLHLA": "PLAX", "PASA": "PSAX", "PMASA": "PSAX", "PMVLSA": "PSAX", "PPMLSA": "PSAX", "A4C": "A4C", "A5C": "A5C", "SC4C": "SC4C", "PMPALA": None}
FAMILIES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]


def main() -> None:
    ap = base_parser("EchoPrime view classifier on EV9V test"); ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frame", default="first", choices=["first", "middle"])
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    repo = paths.repo_root / "third_party" / "EchoPrime"
    sys.path.insert(0, str(repo)); import utils  # noqa: E402  (official EchoPrime utils: COARSE_VIEWS, crop_and_scale)
    assert list(utils.COARSE_VIEWS) == list(TO_FAMILY), "COARSE_VIEWS order changed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11)
    vc.load_state_dict(torch.load(repo / "model_data" / "weights" / "view_classifier.pt", map_location="cpu")); vc.to(device).eval()
    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1); std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1)
    df = load_manifest(paths.manifest_path); df = df[(df["split"] == args.split) & df["frames_dir"].notna()]
    if args.limit:
        df = df.head(args.limit)
    rows = []
    with torch.no_grad():
        for r in df.itertuples(index=False):
            frames = sorted(Path(r.frames_dir).glob("*.jpg"))
            fi = 0 if args.frame == "first" else len(frames) // 2
            img = np.asarray(Image.open(frames[fi]).convert("RGB"))
            x = torch.as_tensor(utils.crop_and_scale(img), dtype=torch.float).permute(2, 0, 1)  # their preprocessing
            x = ((x - mean) / std).unsqueeze(0).to(device)
            logits = vc(x)[0].cpu(); k = int(logits.argmax())
            rows.append({"video_id": r.video_id, "raw_label": r.raw_label, "true_family": EV9V_TO_FAMILY[r.raw_label],
                         "pred_view": utils.COARSE_VIEWS[k], "pred_family": TO_FAMILY[utils.COARSE_VIEWS[k]], "conf": float(torch.softmax(logits, 0)[k])})
    run_dir = new_run_dir(cfg, "echoprime_views")
    import pandas as pd  # noqa: E402
    out = pd.DataFrame(rows); out.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)
    sub = out[out["true_family"].notna()]
    y = [FAMILIES.index(t) for t in sub["true_family"]]; p = [FAMILIES.index(f) if f in FAMILIES else -1 for f in sub["pred_family"]]
    y_arr = np.array(y); p_arr = np.array(p)
    unsupported = float((p_arr < 0).mean())
    p_eval = np.where(p_arr < 0, 5, p_arr)  # unsupported predictions -> extra "other" bucket = always wrong
    summ = classification_summary(y_arr, p_eval, 6)
    conf = pd.crosstab(sub["raw_label"], sub["pred_view"]).to_dict()
    res = {"model": "EchoPrime view_classifier.pt (ConvNeXt-Base, 11 coarse views)", "commit": PINNED, "frame": args.frame, "split": args.split,
           "n_family5": int(len(sub)), "n_all": int(len(out)), "family5_accuracy": summ["accuracy"], "family5_macro_f1_over5": float(np.mean([f for f in _per_class_f1(y_arr, p_eval, 6)[:5]])),
           "share_predicted_unsupported": unsupported, "crosstab_raw_label_x_pred_view": conf,
           "note": "externally pretrained, zero training on EV9V; PSAX levels collapsed; A2C/A3C/SSN/Doppler predictions count as errors"}
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "crosstab_raw_label_x_pred_view"}, indent=1)); print(f"run_dir={run_dir}")


def _per_class_f1(y, p, k):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=list(range(k)), average=None, zero_division=0)


if __name__ == "__main__":
    main()
