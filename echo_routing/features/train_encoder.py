"""Training loop for the ResNet-18 frame classifier (B0 encoder). Resumable; checkpoints every epoch."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from echo_routing.config import gpu_memory_fraction
from echo_routing.ingest.frames import VideoRecord
from echo_routing.features.dataset import EpochFrameDataset, FixedFrameDataset
from echo_routing.evaluate.metrics import classification_summary
from echo_routing.audit.label_map import LabelOntology
from echo_routing.manifest import task_subset
from echo_routing.features.encoder import FrameEncoder, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    task: str
    backbone: str
    image_size: int
    frames_per_video: int
    videos_per_epoch: int | None
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    seed: int
    num_workers: int
    eval_frames_per_video: int
    amp: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        return cls(
            task=d["task"], backbone=d.get("backbone", "resnet18"), image_size=int(d.get("image_size", 224)),
            frames_per_video=int(d.get("frames_per_video", 16)), videos_per_epoch=d.get("videos_per_epoch"),
            batch_size=int(d.get("batch_size", 64)), lr=float(d.get("lr", 3e-4)),
            weight_decay=float(d.get("weight_decay", 1e-4)), epochs=int(d.get("epochs", 30)),
            patience=int(d.get("patience", 6)), seed=int(d.get("seed", 0)),
            num_workers=int(d.get("num_workers", 6)), eval_frames_per_video=int(d.get("eval_frames_per_video", 16)),
            amp=bool(d.get("amp", True)),
        )


def records_from_manifest(df: pd.DataFrame, task: str, split: str) -> list[VideoRecord]:
    sub = task_subset(df, task)
    sub = sub[(sub["split"] == split) & sub["frames_dir"].notna() & (sub["n_frames"] > 0)]
    col = "family5_index" if task == "family5" else "raw9_index"
    return [
        VideoRecord(r.video_id, Path(r.frames_dir), int(r.n_frames), int(getattr(r, col)), split)
        for r in sub.itertuples(index=False)
    ]


def setup_device(seed: int) -> torch.device:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(records: list[VideoRecord], num_classes: int) -> torch.Tensor:
    counts = np.bincount([r.label_index for r in records], minlength=num_classes).astype(float)
    w = counts.sum() / np.maximum(counts, 1.0) / num_classes
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: FrameEncoder, loader: DataLoader, device, num_classes: int, records) -> dict[str, Any]:
    model.eval()
    all_p, all_y, all_v = [], [], []
    for x, y, v in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        all_p.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_y.append(y.numpy()); all_v.append(v.numpy())
    prob = np.concatenate(all_p); y = np.concatenate(all_y); vid = np.concatenate(all_v)
    frame_pred = prob.argmax(1)
    frame = classification_summary(y, frame_pred, num_classes)
    cine_prob = np.zeros((len(records), num_classes)); np.add.at(cine_prob, vid, prob)
    cine_y = np.array([r.label_index for r in records])
    cine = classification_summary(cine_y, cine_prob.argmax(1), num_classes)
    return {"frame": frame, "cine": cine}


def train(cfg: TrainConfig, manifest: pd.DataFrame, ontology: LabelOntology, out_dir: Path,
          resume: Path | None = None) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes = ontology.classes(cfg.task); num_classes = len(classes)
    device = setup_device(cfg.seed)
    train_recs = records_from_manifest(manifest, cfg.task, "train")
    val_recs = records_from_manifest(manifest, cfg.task, "validation")
    if not train_recs or not val_recs:
        raise RuntimeError("no training/validation records with frames on disk; build the manifest first")
    train_ds = EpochFrameDataset(train_recs, cfg.frames_per_video, cfg.image_size, cfg.videos_per_epoch, cfg.seed, augment=True)
    val_ds = FixedFrameDataset(val_recs, cfg.eval_frames_per_video, cfg.image_size)
    dl_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **dl_kw)

    model = FrameEncoder(num_classes, cfg.backbone, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = class_weights(train_recs, num_classes).to(device)
    start_epoch, best_score, bad_epochs = 0, -1.0, 0
    if resume and Path(resume).is_file():
        model, meta = load_checkpoint(resume, map_location=str(device)); model.to(device)
        opt.load_state_dict(meta["optimizer"]); sched.load_state_dict(meta["scheduler"])
        start_epoch, best_score = int(meta["epoch"]) + 1, float(meta.get("best_score", -1.0))
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.resample(epoch); model.train(); t0 = time.time(); losses = []
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), y, weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        metrics = evaluate(model, val_dl, device, num_classes, val_recs)
        score = metrics["cine"]["macro_f1"]
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "seconds": time.time() - t0, "lr": sched.get_last_lr()[0]}
        with log_path.open("a") as fh: fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "seconds")} | {"val_cine_macro_f1": score, "val_frame_acc": metrics["frame"]["accuracy"]}), flush=True)
        extra = {"epoch": epoch, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best_score": max(best_score, score),
                 "classes": list(classes), "task": cfg.task, "image_size": cfg.image_size, "val_metrics": metrics, "config": cfg.__dict__}
        save_checkpoint(out_dir / "last.pt", model, extra)
        if score > best_score:
            best_score, bad_epochs = score, 0; save_checkpoint(out_dir / "best.pt", model, extra)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stop at epoch {epoch} (best cine macro-F1 {best_score:.4f})", flush=True); break
    return out_dir / "best.pt"
