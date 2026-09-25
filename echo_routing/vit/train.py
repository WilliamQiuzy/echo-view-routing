"""Training loop shared by the frame- and clip-level ViT-family models. Resumable; checkpoints every epoch.

Protocol (same as the B0 encoder unless stated): official EV9V train split, class-balanced cine draws, gamma edit,
class-weighted cross-entropy, model selection on validation cine macro-F1 with early stopping. ViT-specific:
AdamW + layer-wise lr decay, linear warm-up then per-step cosine, drop-path, gradient clipping.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from echo_routing.audit.label_map import LabelOntology
from echo_routing.evaluate.metrics import classification_summary
from echo_routing.features.train_encoder import class_weights, records_from_manifest, setup_device
from echo_routing.vit.datasets import EpochClips, EpochFrames, FixedClips, FixedFrames, collate_bytes, soft_targets
from echo_routing.vit.gpu_frames import NORMS, GpuDecoder
from echo_routing.vit.models import ModelSpec, build_model, save_vit_checkpoint
from echo_routing.vit.optim import param_groups, set_lr, warmup_cosine


RESUME_KEYS = ("model", "task", "epochs", "batch_size", "videos_per_epoch", "frames_per_video", "n_frames", "step",
               "lr", "layer_decay", "weight_decay", "warmup_epochs", "seed", "limit_records", "mix_prob", "mix_target")


@dataclass(frozen=True)
class ViTTrainConfig:
    model: str
    task: str
    image_size: int = 224
    frames_per_video: int = 16
    n_frames: int = 16
    step: int = 3
    videos_per_epoch: int | None = None
    batch_size: int = 64
    lr: float = 1e-4
    layer_decay: float = 0.75
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    epochs: int = 30
    patience: int = 6
    seed: int = 0
    num_workers: int = 4
    eval_frames_per_video: int = 16
    eval_clips_per_video: int = 3
    aux_frame_weight: float = 0.5
    mix_prob: float = 0.0                   # transition-aware clips (clip models only)
    mix_target: str = "share"               # share: soft share of frames per view | centre: label of the centre frame
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    class_weighted_loss: bool = True
    limit_records: int | None = None        # smoke runs only
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ViTTrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown ViT training keys: {sorted(unknown)}")
        if d.get("mix_target", "share") not in ("share", "centre"):
            raise ValueError(f"mix_target must be share or centre, got {d['mix_target']!r}")
        return cls(**d)


def _vit_trunk(model) -> tuple[str | None, int]:
    for prefix in ("vit", "spatial"):
        trunk = getattr(model, prefix, None)
        if trunk is not None and hasattr(trunk, "blocks"):
            return prefix, len(trunk.blocks)
    return None, 0


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Class-weighted CE against probability targets, normalised like the hard-label weighted mean
    (sum of w_c * y_c * -log p_c over the batch / sum of w_c * y_c), so one-hot targets give exactly F.cross_entropy."""
    w = target if weight is None else target * weight[None]
    return -(w * F.log_softmax(logits, dim=1)).sum() / w.sum().clamp_min(1e-12)


def _forward_loss(model, kind: str, x: torch.Tensor, y: torch.Tensor, frame_labels: torch.Tensor, t: int, weight,
                  cfg: ViTTrainConfig, num_classes: int) -> torch.Tensor:
    if kind == "frame":
        return F.cross_entropy(model(x).float(), y, weight=weight, label_smoothing=cfg.label_smoothing)
    clips = x.view(y.shape[0], t, *x.shape[1:])
    clip_logits, frame_logits = model(clips)
    if cfg.mix_prob > 0 and cfg.mix_target == "share":    # transition-aware: target = share of frames per view
        loss = soft_cross_entropy(clip_logits.float(), soft_targets(frame_labels, num_classes), weight)
    elif cfg.mix_prob > 0 and cfg.mix_target == "centre":  # transition-aware: the view at the window centre, the
        # sample that inherits this window's probabilities when a stream is routed (window t//2 = its centre sample)
        loss = F.cross_entropy(clip_logits.float(), frame_labels[:, t // 2], weight=weight, label_smoothing=cfg.label_smoothing)
    else:
        loss = F.cross_entropy(clip_logits.float(), y, weight=weight, label_smoothing=cfg.label_smoothing)
    if frame_logits is not None and cfg.aux_frame_weight > 0:   # every frame against its own label
        loss = loss + cfg.aux_frame_weight * F.cross_entropy(frame_logits.float().flatten(0, 1), frame_labels.flatten(),
                                                             weight=weight, label_smoothing=cfg.label_smoothing)
    return loss


@torch.no_grad()
def evaluate(model, loader: DataLoader, decoder: GpuDecoder, kind: str, num_classes: int, records) -> dict[str, Any]:
    """Per-item accuracy and cine-level metrics (mean softmax over the cine's frames or clips)."""
    model.eval()
    probs, ys, vids = [], [], []
    for b in loader:
        x = decoder(b.datas, b.gammas)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            logits = model(x) if kind == "frame" else model(x.view(b.labels.shape[0], b.t, *x.shape[1:]))[0]
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy()); ys.append(b.labels.numpy()); vids.append(b.vids.numpy())
    prob, y, vid = np.concatenate(probs), np.concatenate(ys), np.concatenate(vids)
    cine_prob = np.zeros((len(records), num_classes)); np.add.at(cine_prob, vid, prob)
    cine_y = np.array([r.label_index for r in records])
    return {"item": classification_summary(y, prob.argmax(1), num_classes),
            "cine": classification_summary(cine_y, cine_prob.argmax(1), num_classes)}


def _datasets(cfg: ViTTrainConfig, kind: str, train_recs, val_recs):
    if kind == "frame":
        return (EpochFrames(train_recs, cfg.frames_per_video, cfg.videos_per_epoch, cfg.seed),
                FixedFrames(val_recs, cfg.eval_frames_per_video))
    return (EpochClips(train_recs, cfg.n_frames, cfg.step, cfg.videos_per_epoch, cfg.seed, mix_prob=cfg.mix_prob),
            FixedClips(val_recs, cfg.n_frames, cfg.step, cfg.eval_clips_per_video))


def train(cfg: ViTTrainConfig, manifest: pd.DataFrame, ontology: LabelOntology, out_dir: Path,
          resume: Path | None = None) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    classes = tuple(ontology.classes(cfg.task)); num_classes = len(classes)
    device = setup_device(cfg.seed)
    train_recs = records_from_manifest(manifest, cfg.task, "train")
    val_recs = records_from_manifest(manifest, cfg.task, "validation")
    if cfg.limit_records:
        train_recs, val_recs = train_recs[: cfg.limit_records], val_recs[: max(8, cfg.limit_records // 4)]
    spec = ModelSpec(cfg.model, num_classes, cfg.task, classes, cfg.image_size, cfg.n_frames, cfg.step, extra=dict(cfg.extra))
    kind = spec.kind
    train_ds, val_ds = _datasets(cfg, kind, train_recs, val_recs)
    dl_kw = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0, collate_fn=collate_bytes)
    per_item = 1 if kind == "frame" else cfg.n_frames
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=max(1, (256 // per_item)), shuffle=False, **dl_kw)
    decoder = GpuDecoder(cfg.image_size, NORMS[spec.norm], device)

    model = build_model(spec, pretrained=True).to(device)
    prefix, n_layers = _vit_trunk(model)
    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.weight_decay, cfg.layer_decay if prefix else 1.0, prefix, n_layers),
                            lr=cfg.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl); total = steps_per_epoch * cfg.epochs
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    weight = class_weights(train_recs, num_classes).to(device) if cfg.class_weighted_loss else None
    start_epoch, best, bad, step = 0, -1.0, 0, 0
    if resume and Path(resume).is_file():
        # load into the existing model so the optimiser keeps pointing at the parameters being trained
        meta = torch.load(resume, map_location=str(device), weights_only=False)
        changed = {k for k in RESUME_KEYS if meta["config"].get(k) != dataclasses.asdict(cfg).get(k)}
        if changed:   # the per-step schedule and sampler depend on these; a silent change would alter the protocol
            raise ValueError(f"resume config differs from the checkpoint in {sorted(changed)}")
        model.load_state_dict(meta["state_dict"]); opt.load_state_dict(meta["optimizer"])
        start_epoch, best, bad, step = int(meta["epoch"]) + 1, float(meta["best_score"]), int(meta["bad_epochs"]), int(meta["step"])
    log_path = out_dir / "train_log.jsonl"
    print(json.dumps({"spec": dataclasses.asdict(spec), "train_cines": len(train_recs), "val_cines": len(val_recs),
                      "steps_per_epoch": steps_per_epoch, "vit_trunk": prefix, "layers": n_layers}), flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.resample(epoch); model.train(); t0 = time.time(); losses = []
        for b in train_dl:
            x = decoder(b.datas, b.gammas)
            y, fl = b.labels.to(device, non_blocking=True), b.frame_labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = _forward_loss(model, kind, x, y, fl, b.t, weight, cfg, num_classes)
            set_lr(opt, cfg.lr, warmup_cosine(step, total, warmup))
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); step += 1; losses.append(loss.item())
        metrics = evaluate(model, val_dl, decoder, kind, num_classes, val_recs)
        score = float(metrics["cine"]["macro_f1"])
        improved = score > best
        best, bad = (score, 0) if improved else (best, bad + 1)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics, "seconds": time.time() - t0,
               "lr_factor": warmup_cosine(max(step - 1, 0), total, warmup)}
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"epoch": epoch, "train_loss": rec["train_loss"], "seconds": round(rec["seconds"], 1),
                          "val_cine_macro_f1": score, "val_cine_acc": metrics["cine"]["accuracy"],
                          "val_item_acc": metrics["item"]["accuracy"], "best": best}), flush=True)
        extra = {"epoch": epoch, "optimizer": opt.state_dict(), "best_score": best, "bad_epochs": bad, "step": step,
                 "val_metrics": metrics, "config": dataclasses.asdict(cfg)}
        save_vit_checkpoint(out_dir / "last.pt", model, spec, extra)
        if improved:
            save_vit_checkpoint(out_dir / "best.pt", model, spec, extra)
        elif bad >= cfg.patience:
            print(f"early stop at epoch {epoch} (best val cine macro-F1 {best:.4f})", flush=True)
            break
    return out_dir / "best.pt"
