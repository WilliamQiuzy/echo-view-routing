"""Training loop for the ResNet-18 frame classifier (B0 encoder). Resumable; checkpoints every epoch.

Training workers are restarted every epoch (persistent_workers=False) so that each epoch's resampled frames reach them.
With `val_clip_ids` (a protocol's checkpoint-selection clips) validation uses every 10 Hz sample and the clip-level
majority vote of the final evaluation; without it the legacy rule (16 spaced frames, summed probabilities) is kept."""
from __future__ import annotations

import hashlib
import json
import random
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
from echo_routing.features.dataset import EpochFrameDataset, FixedFrameDataset, RateFrameDataset
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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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


@torch.no_grad()
def evaluate_vote(model: FrameEncoder, loader: DataLoader, device, num_classes: int, records) -> dict[str, Any]:
    """Per-sample argmax over every 10 Hz sample; clip label = majority vote (ties to the lower class index)."""
    model.eval()
    preds, ys, vids = [], [], []
    for x, y, v in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        preds.append(logits.float().argmax(1).cpu().numpy()); ys.append(y.numpy()); vids.append(v.numpy())
    pred = np.concatenate(preds); y = np.concatenate(ys); vid = np.concatenate(vids)
    votes = np.zeros((len(records), num_classes)); np.add.at(votes, (vid, pred), 1)
    cine_y = np.array([r.label_index for r in records])
    return {"frame": classification_summary(y, pred, num_classes), "cine": classification_summary(cine_y, votes.argmax(1), num_classes),
            "rule": "10 Hz samples, clip majority vote"}


def train_loader_kwargs(num_workers: int, cuda: bool) -> dict[str, Any]:
    """Fresh workers every epoch, so EpochFrameDataset.resample() reaches them."""
    return dict(num_workers=num_workers, pin_memory=cuda, persistent_workers=False)


def param_digest(model: torch.nn.Module) -> str:
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def train(cfg: TrainConfig, manifest: pd.DataFrame, ontology: LabelOntology, out_dir: Path,
          resume: Path | None = None, val_clip_ids: list[str] | None = None) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes = ontology.classes(cfg.task); num_classes = len(classes)
    device = setup_device(cfg.seed)
    train_recs = records_from_manifest(manifest, cfg.task, "train")
    val_recs = records_from_manifest(manifest, cfg.task, "validation")
    if val_clip_ids is not None:
        keep = set(val_clip_ids); val_recs = [r for r in val_recs if r.video_id in keep]
        if len(val_recs) != len(keep):
            raise RuntimeError(f"protocol lists {len(keep)} validation clips, {len(val_recs)} found in the manifest")
    if not train_recs or not val_recs:
        raise RuntimeError("no training/validation records with frames on disk; build the manifest first")
    train_ds = EpochFrameDataset(train_recs, cfg.frames_per_video, cfg.image_size, cfg.videos_per_epoch, cfg.seed, augment=True)
    val_ds = RateFrameDataset(val_recs, cfg.image_size) if val_clip_ids is not None else FixedFrameDataset(val_recs, cfg.eval_frames_per_video, cfg.image_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          **train_loader_kwargs(cfg.num_workers, device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    run_eval = evaluate_vote if val_clip_ids is not None else evaluate

    model = FrameEncoder(num_classes, cfg.backbone, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = class_weights(train_recs, num_classes).to(device)
    init_sha = param_digest(model)
    (out_dir / "init_digest.txt").write_text(f"seed {cfg.seed}\ninit_sha256 {init_sha}\n")
    start_epoch, best_score, bad_epochs = 0, -1.0, 0
    if resume and Path(resume).is_file():
        model, meta = load_checkpoint(resume, map_location=str(device)); model.to(device)
        opt.load_state_dict(meta["optimizer"]); sched.load_state_dict(meta["scheduler"])
        start_epoch, best_score = int(meta["epoch"]) + 1, float(meta.get("best_score", -1.0))
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.resample(epoch); model.train(); t0 = time.time(); losses = []; order = hashlib.sha256(); seen = set()
        for x, y, v, f in train_dl:
            if epoch < 2:  # sample-order digest from what the workers actually returned
                order.update(np.stack([v.numpy(), f.numpy()]).astype(np.int64).tobytes()); seen.update(zip(v.tolist(), f.tolist()))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), y, weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        metrics = run_eval(model, val_dl, device, num_classes, val_recs)
        score = metrics["cine"]["macro_f1"]
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "seconds": time.time() - t0, "lr": sched.get_last_lr()[0],
               "init_sha256": init_sha}
        if epoch < 2:
            rec["sample_order_sha256"] = order.hexdigest(); rec["distinct_samples"] = len(seen)
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


def train_fixed_epochs(cfg: TrainConfig, train_recs: list[VideoRecord], classes: list[str], out_dir: Path,
                       resume: Path | None = None, pretrained: bool = True) -> Path:
    """The B0 recipe of `train` on the given records for exactly cfg.epochs epochs, with no validation data, early
    stopping or checkpoint selection. last.pt is saved every epoch (resumable) and returned."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not train_recs:
        raise RuntimeError("no training records")
    num_classes = len(classes); device = setup_device(cfg.seed)
    train_ds = EpochFrameDataset(train_recs, cfg.frames_per_video, cfg.image_size, cfg.videos_per_epoch, cfg.seed, augment=True)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          **train_loader_kwargs(cfg.num_workers, device.type == "cuda"))
    model = FrameEncoder(num_classes, cfg.backbone, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weights = class_weights(train_recs, num_classes).to(device)
    init_sha = param_digest(model); start_epoch = 0
    if resume and Path(resume).is_file():
        model, meta = load_checkpoint(resume, map_location=str(device)); model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay); opt.load_state_dict(meta["optimizer"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs); sched.load_state_dict(meta["scheduler"])
        start_epoch, init_sha = int(meta["epoch"]) + 1, meta.get("init_sha256", init_sha)
    else:
        (out_dir / "init_digest.txt").write_text(f"seed {cfg.seed}\ninit_sha256 {init_sha}\n")
        (out_dir / "train_log.jsonl").unlink(missing_ok=True)  # a fresh run starts a fresh log
    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.resample(epoch); model.train(); t0 = time.time(); losses = []
        for x, y, _, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), y, weight=weights)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.time() - t0, "lr": sched.get_last_lr()[0],
               "n_train_records": len(train_recs), "n_samples": len(train_ds), "init_sha256": init_sha}
        with log_path.open("a") as fh: fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("epoch", "train_loss", "seconds")}), flush=True)
        save_checkpoint(out_dir / "last.pt", model, {"epoch": epoch, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "classes": list(classes),
                                                     "task": cfg.task, "image_size": cfg.image_size, "config": cfg.__dict__, "init_sha256": init_sha,
                                                     "selection": "none: the last epoch is kept"})
    return out_dir / "last.pt"
