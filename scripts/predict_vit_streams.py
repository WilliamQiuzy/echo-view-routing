#!/usr/bin/env python
"""Per-stream five-family probabilities (ladder lookup files) and cine-level metrics for a ViT-family checkpoint.

vit_frame : per-sample frame-head probabilities from its feature cache (no temporal processing, like EchoPrime/B0).
vivit_fe  : cached frame embeddings -> temporal Transformer on sliding windows (16 samples = 1.6 s, stride 5).
mvit_*    : the same sliding windows over decoded frames (joint space-time model, needs pixels).
Streams come from plans.json (identical to every other method). Nine-code outputs are mapped to family5 exactly as
for STFM/EchoViewCLIP (PMPALA mass dropped). Writes cache/vit_streams/<tag>/<stream_id>.npz (prob T x 5, raw T x C)
and <run_dir>/video_level.json + video_level_<split>.csv: cine probability = mean over the cine's windows (clip
models) or samples (frame model); nine-code metrics over all official cines, family5 metrics over family5 cines.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.metrics_classification import classification_summary  # noqa: E402
from echo_routing.features.cache import load_video_features, variant_dir  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash  # noqa: E402
from echo_routing.features.sampling import stride_indices  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.manifest import load_manifest, task_subset  # noqa: E402
from echo_routing.temporal.windowed import FAMILY5, RAW9_TO_FAMILY5, map_probs  # noqa: E402
from echo_routing.vit.gpu_frames import NORMS, GpuDecoder, read_bytes  # noqa: E402
from echo_routing.vit.infer import WINDOW, assemble, windowed_probs  # noqa: E402
from echo_routing.vit.models import load_vit_checkpoint  # noqa: E402

WIN_BATCH = {"emb": 256, "pix": 16}   # windows per forward (pixel windows are 16 decoded frames each)


def to_family5(raw: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    if list(classes) == FAMILY5:
        return raw
    return map_probs(raw, classes, RAW9_TO_FAMILY5)


class Predictor:
    def __init__(self, model, spec, device, cache_root: Path, feat_hash: str | None, frames_root: Path) -> None:
        self.model, self.spec, self.device = model, spec, device
        self.cache_root, self.feat_hash, self.frames_root = cache_root, feat_hash, frames_root
        self._vf: dict = {}; self._paths: dict = {}
        self.decoder = GpuDecoder(spec.image_size, NORMS[spec.norm], device)

    def loader(self, video_id: str, variant: str):
        key = (video_id, variant)
        if key not in self._vf:
            if len(self._vf) > 4000:
                self._vf.clear()
            self._vf[key] = load_video_features(variant_dir(self.cache_root, self.feat_hash, variant) / f"{video_id}.npz")
        return self._vf[key]

    def _frames(self, refs) -> torch.Tensor:
        datas = []
        for vid, fi, _ in refs:
            if vid not in self._paths:
                self._paths[vid] = list_frame_paths(self.frames_root / vid)
            p = self._paths[vid]; datas.append(read_bytes(p[min(int(fi), len(p) - 1)]))
        out = [self.decoder(datas[s:s + 512], [float(r[2]) for r in refs[s:s + 512]]) for s in range(0, len(refs), 512)]
        return torch.cat(out)

    def _windows_fn(self, seq: torch.Tensor, kind: str):
        @torch.no_grad()
        def fn(wins):
            probs = []
            step = WIN_BATCH[kind]
            for s in range(0, len(wins), step):
                idx = torch.as_tensor(np.stack(wins[s:s + step]), device=seq.device)
                x = seq[idx]                                   # (W, 16, D) embeddings or (W, 16, 3, H, W) frames
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    lg = self.model.temporal_logits(x) if kind == "emb" else self.model(x)[0]
                probs.append(torch.softmax(lg.float(), 1).cpu().numpy())
            return np.concatenate(probs)
        return fn

    def stream(self, refs) -> tuple[np.ndarray, np.ndarray]:
        """(per-sample raw class probabilities, per-window probabilities or per-sample for frame models)."""
        n = len(refs)
        if self.spec.name == "vit_frame":
            p = assemble(refs, self.loader, "prob"); return p, p
        if self.spec.name == "vivit_fe":
            emb = torch.as_tensor(assemble(refs, self.loader, "feat"), dtype=torch.float32, device=self.device)
            return windowed_probs(n, self._windows_fn(emb, "emb"))
        frames = self._frames(refs)
        return windowed_probs(n, self._windows_fn(frames, "pix"))


def video_refs(row, target_hz: float) -> list:
    return [[row.video_id, int(f), 1.0] for f in stride_indices(int(row.n_frames), float(row.fps_playback), target_hz)]


def main() -> None:
    ap = base_parser("ViT-family stream predictions + cine-level metrics")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--plans", default="data/window_plans/79f4a41af6e3/plans.json")
    ap.add_argument("--feat-hash", default=None, help="feature cache of this checkpoint (frame / vivit); default = its hash")
    ap.add_argument("--tag", default=None); ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    os.environ.setdefault("HF_HOME", str(paths.data_root / "hf_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    ck = Path(args.ckpt); model, spec, meta = load_vit_checkpoint(ck); model.eval().to(device)
    if spec.kind == "clip" and (spec.n_frames != WINDOW or spec.step != 3):
        raise SystemExit(f"{spec.name}: trained on {spec.n_frames} frames every {spec.step}, but evaluation windows are "
                         f"{WINDOW} samples on the 10 Hz grid (step 3); retrain or change the window")
    h = checkpoint_hash(ck); tag = args.tag or f"{spec.name}_{h}"
    pred = Predictor(model, spec, device, paths.cache_root, args.feat_hash or h, paths.ev9v_images)
    run_dir = new_run_dir(cfg, f"predict_{spec.name}")
    out = paths.cache_root / "vit_streams" / tag; out.mkdir(parents=True, exist_ok=True)
    plans = json.loads((paths.repo_root / args.plans).read_text())["streams"]
    items = list(plans.items())[: args.limit] if args.limit else list(plans.items())
    for n_done, (sid, pl) in enumerate(items, 1):
        raw, _ = pred.stream(pl["refs"])
        np.savez_compressed(out / f"{sid}.npz", prob=to_family5(raw, spec.classes).astype(np.float32), raw=raw.astype(np.float32))
        if n_done % 250 == 0:
            print(f"  streams {n_done}/{len(items)}", flush=True)
    # cine level on the official splits (nine-code superset for raw9 models)
    df = task_subset(load_manifest(paths.manifest_path), spec.task)
    df = df[df["frames_dir"].notna() & (df["n_frames"] > 0)]
    report = {"ckpt": str(ck), "ckpt_hash": h, "model": spec.name, "task": spec.task, "classes": list(spec.classes),
              "pred_dir": str(out), "val_metrics_at_selection": meta.get("val_metrics"), "epoch": meta.get("epoch")}
    for split in args.splits.split(","):
        rows = list(df[df["split"] == split].itertuples(index=False))[: args.limit]
        probs = []
        for r in rows:
            _, per = pred.stream(video_refs(r, cfg["sampling"]["target_hz"]))
            probs.append(per.mean(0))
        probs = np.stack(probs); label_col = "family5_index" if spec.task == "family5" else "raw9_index"
        y = np.array([int(getattr(r, label_col)) for r in rows])
        tab = pd.DataFrame(probs, columns=[f"p_{c}" for c in spec.classes])
        tab.insert(0, "label", y); tab.insert(0, "raw_label", [r.raw_label for r in rows]); tab.insert(0, "video_id", [r.video_id for r in rows])
        tab.to_csv(run_dir / f"video_level_{split}.csv", index=False)
        res = {"native_task": classification_summary(y, probs.argmax(1), len(spec.classes))}
        fam = to_family5(probs, spec.classes); fam_y = np.array([RAW9_TO_FAMILY5.get(r.raw_label, None) for r in rows], dtype=object)
        keep = np.array([v is not None for v in fam_y])
        res["family5"] = classification_summary(fam_y[keep].astype(int), fam[keep].argmax(1), len(FAMILY5))
        report[split] = res
        print(json.dumps({split: {k: {m: v[m] for m in ("n", "accuracy", "macro_f1", "balanced_accuracy")} for k, v in res.items()}}), flush=True)
    (run_dir / "video_level.json").write_text(json.dumps(report, indent=1))
    print(f"pred_dir={out}\nrun_dir={run_dir}")


if __name__ == "__main__":
    main()
