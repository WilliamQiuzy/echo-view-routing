#!/usr/bin/env python
"""10 Hz frame probabilities of B0 and EchoPrime for EV9V validation/test clips, with frames in true temporal order.

Written for round 1 (2026-09-30), when the shared caches still read frames in name order (see
docs/results/2026-09-30_realvideo_round1). Output: <out>/probs_sorted/<video_id>.npz with frame_idx (true 30-fps frame
number), t_sec, prob_b0, prob_ep (a sixth "other" column) and raw_ep.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.encoder import load_checkpoint  # noqa: E402
from echo_routing.features.sampling import stride_indices  # noqa: E402
from echo_routing.ingest.frames import letterbox, to_tensor_array  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.windowed import ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, map_probs  # noqa: E402

B0_CKPT = "checkpoints/20260913-033411-b0_frame_encoder-b072725b-s0/best.pt"
EP_MEAN = np.array([29.110628, 28.076836, 29.096405], dtype=np.float32).reshape(3, 1, 1)
EP_STD = np.array([47.989223, 46.456997, 47.20083], dtype=np.float32).reshape(3, 1, 1)
CROP_AND_SCALE = None


class Clips(Dataset):
    def __init__(self, rows: list[tuple[str, str, float]], size: int) -> None:
        self.rows, self.size = rows, size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        vid, frames_dir, fps = self.rows[i]
        paths = list_frame_paths(Path(frames_dir))
        idx = stride_indices(len(paths), fps, 10.0)
        ims = [Image.open(paths[k]).convert("RGB") for k in idx]
        ep = np.stack([(CROP_AND_SCALE(np.asarray(im)).astype(np.float32).transpose(2, 0, 1) - EP_MEAN) / EP_STD for im in ims])
        b0 = np.stack([to_tensor_array(letterbox(im, self.size)) for im in ims])
        return {"vid": vid, "idx": idx, "fps": fps, "b0": torch.from_numpy(b0), "ep": torch.from_numpy(ep)}


def _one_thread(_: int) -> None:
    torch.set_num_threads(1)


def main() -> None:
    global CROP_AND_SCALE
    ap = argparse.ArgumentParser(description="Temporally ordered B0/EchoPrime probabilities for EV9V val/test")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(); paths = load_paths(); out = Path(args.out) / "probs_sorted"; out.mkdir(parents=True, exist_ok=True)
    repo = paths.repo_root / "third_party" / "EchoPrime"; cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import utils  # noqa: E402  (official EchoPrime crop_and_scale)
    os.chdir(cwd); CROP_AND_SCALE = utils.crop_and_scale
    man = pd.read_csv(paths.manifest_path)
    man = man[(man["dataset"] == "EV9V") & man["split"].isin(["validation", "test"]) & man["family5_label"].notna()]
    rows = [(r.video_id, r.frames_dir, float(r.fps_playback)) for r in man.itertuples(index=False)
            if not (out / f"{r.video_id}.npz").is_file()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    b0, meta = load_checkpoint(paths.repo_root / B0_CKPT); b0 = b0.to(device).eval()
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11)
    vc.load_state_dict(torch.load(paths.repo_root / "models_frozen" / "echoprime_view_classifier" / "release_v1.0.0" / "view_classifier.pt", map_location="cpu"))
    vc = vc.to(device).eval()
    dl = DataLoader(Clips(rows, int(meta.get("image_size", 224))), batch_size=None, num_workers=args.workers, worker_init_fn=_one_thread)
    with torch.no_grad():
        for k, it in enumerate(dl, 1):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pb = torch.softmax(b0(it["b0"].to(device)).float(), 1).cpu().numpy()
                raw = torch.softmax(vc(it["ep"].to(device)).float(), 1).cpu().numpy()
            fam = map_probs(raw, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5)
            idx = np.asarray(it["idx"])  # the loader converts arrays to tensors
            np.savez_compressed(out / f"{it['vid']}.npz", frame_idx=idx, t_sec=(idx / float(it["fps"])).astype(np.float32),
                                prob_b0=pb.astype(np.float32), prob_ep=np.c_[fam, np.clip(1 - fam.sum(1), 0, None)].astype(np.float32),
                                raw_ep=raw.astype(np.float32))
            if k % 200 == 0:
                print(f"  {k}/{len(rows)}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
