#!/usr/bin/env python
"""Predict extra streams (e.g. demo recipes) with the frozen official MS-TCN and ASFormer models, using the authors'
model classes and the same zero-padded 2048-d feature convention as the training export. Appends npz files to the
existing prediction dirs so the registry lookups (b4/b5) cover the extra stream ids."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, variant_dir  # noqa: E402

FEATURES_DIM = 2048


def main() -> None:
    ap = base_parser("Official MS-TCN/ASFormer on extra recipes"); ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--recipes", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    recipes = [recipe_from_dict(d) for d in json.loads(Path(args.recipes).read_text())]
    streams = {r.recipe_id: render(r, loader) for r in recipes}
    n_classes = next(iter(streams.values())).prob.shape[1]

    def padded(feat):
        x = np.zeros((FEATURES_DIM, feat.shape[0]), dtype=np.float32); x[: feat.shape[1]] = feat.T
        return torch.from_numpy(x).unsqueeze(0).to(device)

    # MS-TCN (official model class)
    sys.path.insert(0, str(paths.repo_root / "third_party" / "ms-tcn"))
    from model import MultiStageModel  # noqa: E402
    ms = MultiStageModel(4, 10, 64, FEATURES_DIM, n_classes).to(device)
    ms.load_state_dict(torch.load(paths.repo_root / "models_frozen" / "mstcn_official" / "v1" / "epoch-50.model", map_location=device)); ms.eval()
    out_ms = paths.cache_root / "mstcn_official" / args.ckpt_hash
    # ASFormer (official trainer/model class)
    sys.modules.pop("model", None); sys.path.insert(0, str(paths.repo_root / "third_party" / "ASFormer"))
    from model import Trainer  # noqa: E402
    tr = Trainer(10, 2, 2, 64, FEATURES_DIM, n_classes, 0.3); tr.model.to(device)
    tr.model.load_state_dict(torch.load(paths.repo_root / "models_frozen" / "asformer_official" / "v1" / "epoch-120.model", map_location=device)); tr.model.eval()
    out_as = paths.cache_root / "asformer_official" / args.ckpt_hash
    with torch.no_grad():
        for sid, s in streams.items():
            x = padded(s.feat); T = x.shape[2]
            p_ms = torch.softmax(ms(x, torch.ones(1, n_classes, T, device=device))[-1], dim=1)[0].T.cpu().numpy()
            p_as = torch.softmax(tr.model(x, torch.ones(x.size(), device=device))[-1], dim=1)[0].T.cpu().numpy()
            np.savez_compressed(out_ms / f"{sid}.npz", prob=p_ms.astype(np.float32)); np.savez_compressed(out_as / f"{sid}.npz", prob=p_as.astype(np.float32))
    print(f"predicted {len(streams)} streams -> {out_ms}, {out_as}")


if __name__ == "__main__":
    main()
