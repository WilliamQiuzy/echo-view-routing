"""Shared training loop of the round-1 temporal heads (2026-10-02 plan, Tasks 3-4).

The same for every head: AdamW, linear warm-up then cosine decay per update, gradient clipping, the deterministic
batch plan of TrainData, checkpoint selection on the selection half (dev_metrics.select_checkpoint), and early stopping
once the selected epoch is `patience` epochs old. Every epoch logs each loss term, the selection-half metrics, the
predicted boundary positive rate and gate mean (when the head has them), peak GPU memory and time, and keeps the
epoch weights so the rule can return to any earlier epoch.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_checkpoint
from echo_routing.temporal.dev_data import EvalSet, TrainData, collate

LOGGED_METRICS = ("native_macro_f1", "native_accuracy", "native_frame_accuracy", "native_internal_cuts_per_clip",
                  "native_internal_cuts_per_min", "segment_f1_50_multi", "boundary_recall_multi", "boundary_precision_multi",
                  "boundary_f1_multi", "same_class_cut_rate_multi", "multi_frame_accuracy", "segment_f1_50_pairs",
                  "boundary_recall_pairs")


@dataclass(frozen=True)
class Settings:
    seed: int
    lr: float
    weight_decay: float
    streams_per_batch: int
    natives_per_batch: int
    grad_clip: float
    warmup_epochs: int
    max_epochs: int
    patience: int
    native_window: int
    native_per_epoch: int

    @classmethod
    def from_cfg(cls, tc: dict, seed: int, lr: float | None = None) -> "Settings":
        bs = int(tc["batch_size"])
        return cls(seed=int(seed), lr=float(lr if lr is not None else tc["lr"]), weight_decay=float(tc["weight_decay"]),
                   streams_per_batch=bs // 2, natives_per_batch=bs - bs // 2, grad_clip=float(tc["grad_clip"]),
                   warmup_epochs=int(tc["warmup_epochs"]), max_epochs=int(tc["max_epochs"]), patience=int(tc["patience"]),
                   native_window=int(tc["native_window"]), native_per_epoch=int(tc["native_per_epoch"]))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def n_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def decoded_digest(decoded: dict[str, dict[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for k in sorted(decoded):
        h.update(k.encode()); h.update(np.asarray(decoded[k]["labels"], np.int64).tobytes())
    return h.hexdigest()


def diagnostics(decoded: dict[str, dict[str, np.ndarray]], ev: EvalSet) -> dict:
    """Predicted boundary positive rate (b > 0.5 over transitions 1..n-1) and gate mean, on natives and multi streams."""
    out = {}
    for part, ids in (("native", [s.seq_id for s in ev.natives]), ("multi", [s.recipe.recipe_id for s in ev.multi])):
        rows = [decoded[i] for i in ids]
        if rows and "boundary_prob" in rows[0]:
            b = np.concatenate([r["boundary_prob"][1:] for r in rows])
            out[f"boundary_pos_rate_{part}"] = float((b > 0.5).mean()) if b.size else None
        if rows and "gate" in rows[0]:
            out[f"gate_mean_{part}"] = float(np.concatenate([r["gate"] for r in rows]).mean())
    return out


def evaluate_head(head, model: torch.nn.Module, ev: EvalSet, hz: float, tol_s: float, device) -> tuple[dict, dict, dict]:
    decoded = head.decode(model, ev.sequences(), device)
    metrics, tables = evaluate_predictions({k: v["labels"] for k, v in decoded.items()}, ev, hz, tol_s)
    return {**metrics, **diagnostics(decoded, ev)}, tables, decoded


def train_head(head, data: TrainData, sel: EvalSet, st: Settings, ckpt_dir: Path, floor: float, tie: float, hz: float,
               tol_s: float, device, log: Callable[[str], None] = print, augment: Callable | None = None) -> dict:
    """Train one head; returns the epoch records, the chosen epoch, its status and paths."""
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(st.seed)
    model = head.build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=st.lr, weight_decay=st.weight_decay)
    steps = len(data.epoch_batches(st.seed, 0, st.native_per_epoch, st.native_window, st.streams_per_batch, st.natives_per_batch))
    total, warmup = steps * st.max_epochs, steps * st.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total))
    records, updates, chosen = [], 0, None
    log_path = ckpt_dir / "train_log.jsonl"; log_path.write_text("")
    for epoch in range(st.max_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time(); model.train(); sums = defaultdict(float); n_b = 0
        batches = data.epoch_batches(st.seed, epoch, st.native_per_epoch, st.native_window, st.streams_per_batch, st.natives_per_batch)
        aug_rng = np.random.default_rng([st.seed, epoch, 7])
        for seqs in batches:
            if augment is not None:
                seqs = [augment(s, aug_rng) for s in seqs]
            batch = collate(seqs, device)
            loss, parts = head.loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, update {updates}: {parts}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), st.grad_clip)
            opt.step(); sched.step(); updates += 1; n_b += 1
            sums["total"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] += v
        t_train = time.time() - t0
        metrics, _, _ = evaluate_head(head, model, sel, hz, tol_s, device)
        rec = {"epoch": epoch, "updates": updates, "lr": float(sched.get_last_lr()[0]),
               "loss": {k: v / n_b for k, v in sums.items()}, "seconds_train": t_train, "seconds_eval": time.time() - t0 - t_train,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
               **{k: metrics[k] for k in LOGGED_METRICS},
               **{k: v for k, v in metrics.items() if k.startswith(("boundary_pos_rate", "gate_mean"))}}
        records.append(rec)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        sel_res = select_checkpoint(records, floor, tie); chosen = sel_res["chosen"]["epoch"]
        rec["chosen_epoch"] = chosen
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"epoch {epoch} upd {updates} loss {rec['loss']['total']:.4f} | native F1 {rec['native_macro_f1']:.4f} "
            f"segF1 {rec['segment_f1_50_multi']:.4f} rec {rec['boundary_recall_multi']:.3f} cuts {rec['native_internal_cuts_per_clip']:.3f} "
            f"| chosen {chosen} ({t_train:.0f}s+{rec['seconds_eval']:.0f}s)")
        if epoch - chosen >= st.patience:
            log(f"early stop: chosen epoch {chosen} is {epoch - chosen} epochs old")
            break
    final = select_checkpoint(records, floor, tie)
    shutil.copyfile(ckpt_dir / f"epoch_{final['chosen']['epoch']:03d}.pt", ckpt_dir / "best.pt")
    shutil.copyfile(ckpt_dir / f"epoch_{records[-1]['epoch']:03d}.pt", ckpt_dir / "last.pt")
    return {"records": records, "chosen_epoch": final["chosen"]["epoch"], "status": final["status"], "floor": floor,
            "updates": updates, "settings": asdict(st), "n_params": n_params(model),
            "best": str(ckpt_dir / "best.pt"), "last": str(ckpt_dir / "last.pt")}


@torch.no_grad()
def time_decode(head, model: torch.nn.Module, ev: EvalSet, device, repeats: int = 3) -> dict:
    """Head-only decoding time per sample over one validation half (features already extracted)."""
    seqs = ev.sequences(); n = sum(s.labels.size for s in seqs); best = math.inf
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter(); head.decode(model, seqs, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        best = min(best, time.perf_counter() - t0)
    return {"head_ms_per_sample": 1000.0 * best / n, "n_samples": int(n), "device": str(device), "includes": "decoding only"}


def overfit_check(head, seqs: list, device, steps: int = 300, lr: float = 1e-3, seed: int = 0,
                  log: Callable[[str], None] = print) -> dict:
    """Small-sample check (Task 3): train on a fixed set of sequences in one batch and record whether the classification
    and boundary losses fall. Dropout stays on during training; accuracy is read in eval mode afterwards."""
    seed_everything(seed)
    model = head.build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    batch = collate(seqs, device); history = []
    for step in range(steps):
        model.train(); loss, parts = head.loss(model, batch)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 25 == 0 or step == steps - 1:
            history.append({"step": step, "total": float(loss.detach()), **parts})
            log(f"overfit step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in history[-1].items() if k != "step"))
    decoded = head.decode(model, seqs, device)
    acc = float(np.mean(np.concatenate([decoded[s.seq_id]["labels"] == s.labels for s in seqs])))
    out = {"n_sequences": len(seqs), "steps": steps, "lr": lr, "history": history, "train_accuracy_eval_mode": acc}
    if "boundary_prob" in next(iter(decoded.values())):
        pos, same = [], []
        for s in seqs:
            b = decoded[s.seq_id]["boundary_prob"]
            for j in s.joins:
                (pos if s.labels[j] != s.labels[j - 1] else same).append(float(b[j]))
        out["boundary_prob_at_changes_mean"] = float(np.mean(pos)) if pos else None
        out["boundary_prob_at_same_joins_mean"] = float(np.mean(same)) if same else None
    return out
