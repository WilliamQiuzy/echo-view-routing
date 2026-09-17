#!/usr/bin/env python
"""B8: EchoPrime released view classifier applied to every 10 Hz sample frame of each stream (their preprocessing),
11 coarse views mapped to family5 (unsupported views' mass dropped). Writes cache/echoprime_windowed/<hash>/<stream_id>.npz."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.ingest.frames import apply_gamma, list_frame_paths  # noqa: E402
from echo_routing.temporal.windowed import ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, map_probs  # noqa: E402


def main() -> None:
    ap = base_parser("EchoPrime per-frame on streams"); ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    repo = paths.repo_root / "third_party" / "EchoPrime"; cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import utils  # noqa: E402  (official: crop_and_scale, COARSE_VIEWS)
    os.chdir(cwd); assert list(utils.COARSE_VIEWS) == ECHOPRIME_VIEWS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11)
    vc.load_state_dict(torch.load(paths.repo_root / "models_frozen" / "echoprime_view_classifier" / "release_v1.0.0" / "view_classifier.pt", map_location="cpu")); vc.to(device).eval()
    mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1); std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1)
    plans = json.loads((paths.data_root / "window_plans" / args.ckpt_hash / "plans.json").read_text())
    frames_root = Path(plans["frames_root"]); out = paths.cache_root / "echoprime_windowed" / args.ckpt_hash; out.mkdir(parents=True, exist_ok=True)
    pcache: dict[str, list] = {}
    items = list(plans["streams"].items())[: args.limit] if args.limit else list(plans["streams"].items())
    with torch.no_grad():
        for n_done, (sid, pl) in enumerate(items, 1):
            if (out / f"{sid}.npz").is_file():
                continue
            xs = []
            for video_id, fi, gamma in pl["refs"]:
                if video_id not in pcache:
                    pcache[video_id] = list_frame_paths(frames_root / video_id)
                with Image.open(pcache[video_id][min(fi, len(pcache[video_id]) - 1)]) as im:
                    im = im.convert("RGB"); im = apply_gamma(im, gamma) if gamma != 1.0 else im
                    x = torch.as_tensor(utils.crop_and_scale(np.asarray(im)), dtype=torch.float).permute(2, 0, 1)
                xs.append((x - mean) / std)
            raw = []
            for i in range(0, len(xs), 64):
                raw.append(torch.softmax(vc(torch.stack(xs[i:i + 64]).to(device)).float(), dim=1).cpu().numpy())
            raw = np.concatenate(raw); fam = map_probs(raw, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5)
            np.savez_compressed(out / f"{sid}.npz", prob=fam.astype(np.float32), raw=raw.astype(np.float32))
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(items)}", flush=True)
    print(f"pred_dir={out}")


if __name__ == "__main__":
    main()
