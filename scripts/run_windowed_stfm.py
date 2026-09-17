#!/usr/bin/env python
"""B6: frozen official STFM applied with a sliding window over stream frames (authors' test procedure per window).

Per window (48 consecutive 30-fps frames = 16 samples, stride 5 samples): their test() rule — 10 key frames spaced
over the window, each with a 5-frame clip at interval 5, Resize(256)/CenterCrop(224)/ImageNet-normalise,
evidence = softplus(logits), alpha = evidence + 0.8, summed over key frames -> probabilities. Nine-code output is mapped
to family5 (PMPALA mass dropped). Writes cache/stfm_windowed/<hash>/<stream_id>.npz (prob T x 5, raw T x 9).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.ingest.frames import apply_gamma, list_frame_paths  # noqa: E402
from echo_routing.temporal.windowed import RAW9_ORDER, RAW9_TO_FAMILY5, assign_to_samples, map_probs, window_centres, window_samples  # noqa: E402

WINDOW, STRIDE, FPS_PER_SAMPLE = 16, 5, 3
NUM_FRAMES, CLIP_LENGTH, CLIP_INTERVAL, LAMB2 = 10, 5, 5, 0.8   # authors' test defaults


def clip_indices(image_num: int) -> tuple[list[int], list[list[int]]]:
    keys = torch.linspace(0, image_num - 1, min(NUM_FRAMES, image_num)).long().tolist()
    keys += [image_num - 1] * (NUM_FRAMES - len(keys))
    half = (CLIP_LENGTH // 2) * CLIP_INTERVAL; clips = []
    for k in keys:
        clips.append([min(max(k - half + i * CLIP_INTERVAL, 0), image_num - 1) for i in range(CLIP_LENGTH)])
    return keys, clips


def main() -> None:
    ap = base_parser("STFM windowed"); ap.add_argument("--ckpt-hash", required=True)
    ap.add_argument("--stfm-ckpt", default="models_frozen/stfm_official/seed100/model.ckpt"); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    repo = paths.repo_root / "third_party" / "stfm"; sys.path.insert(0, str(repo))
    from models_enhanced import create_enhanced_stfm  # noqa: E402  (official)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    net = create_enhanced_stfm(num_classes=9, backbone="resnet18", hidden_size=512, num_layers=2, embed_dims=128)
    ck = torch.load(paths.repo_root / args.stfm_ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"]); net.to(device).eval()
    tf = v2.Compose([v2.Resize(256), v2.CenterCrop(224), v2.ToImage()])
    norm = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    plans = json.loads((paths.data_root / "window_plans" / args.ckpt_hash / "plans.json").read_text())
    frames_root = Path(plans["frames_root"]); out = paths.cache_root / "stfm_windowed" / args.ckpt_hash; out.mkdir(parents=True, exist_ok=True)
    paths_cache: dict[str, list] = {}; img_cache: dict[tuple, torch.Tensor] = {}

    def frame(video_id, fi, gamma):
        key = (video_id, fi, gamma)
        if key not in img_cache:
            if video_id not in paths_cache:
                paths_cache[video_id] = list_frame_paths(frames_root / video_id)
            p = paths_cache[video_id][min(fi, len(paths_cache[video_id]) - 1)]
            with Image.open(p) as im:
                im = im.convert("RGB"); im = apply_gamma(im, gamma) if gamma != 1.0 else im
                img_cache[key] = tf(im)
            if len(img_cache) > 6000:
                img_cache.clear()
        return img_cache[key]

    items = list(plans["streams"].items())[: args.limit] if args.limit else list(plans["streams"].items())
    with torch.no_grad():
        for n_done, (sid, pl) in enumerate(items, 1):
            if (out / f"{sid}.npz").is_file():
                continue
            refs = pl["refs"]; n = pl["n"]; centres = window_centres(n, STRIDE); raws = []
            for c in centres:
                samp = window_samples(int(c), n, WINDOW)
                frames = [(refs[s][0], refs[s][1] + k, refs[s][2]) for s in samp for k in range(FPS_PER_SAMPLE)]
                imgs = torch.stack([frame(*f) for f in frames]).to(device); imgs = norm(imgs)
                keys, clips = clip_indices(len(frames))
                frames_batch = imgs[keys]; clips_batch = torch.stack([imgs[c_] for c_ in clips])
                logits = net((frames_batch, clips_batch))
                alpha = F.softplus(logits) + LAMB2; total = alpha.sum(0, keepdim=True)
                raws.append((total / total.sum()).squeeze(0).cpu().numpy())
            raw = assign_to_samples(n, centres, np.stack(raws)); fam = map_probs(raw, RAW9_ORDER, RAW9_TO_FAMILY5)
            np.savez_compressed(out / f"{sid}.npz", prob=fam.astype(np.float32), raw=raw.astype(np.float32))
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(items)}", flush=True)
    print(f"pred_dir={out}")


if __name__ == "__main__":
    main()
