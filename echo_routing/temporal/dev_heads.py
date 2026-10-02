"""Temporal heads compared in round 1 (2026-10-02 plan, Task 4).

M0 = official MS-TCN, M1 = official ASRF, M2-M4 = the boundary-controlled context model and its ablations. Each head
provides build(), loss(model, batch) -> (loss, parts) and decode(model, seqs, device) -> per-sequence outputs. The
official heads import the authors' classes and losses from the pinned checkouts without modification; the shared
training recipe (data, optimiser, schedule, update budget, checkpoint rule) is the task adaptation.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.temporal.dev_data import Seq, collate
from echo_routing.temporal.losses.boundary_context import total_loss
from echo_routing.temporal.models.boundary_context import BoundaryContextConfig, BoundaryContextNet

MSTCN_PINNED = "33ed91c0c7576650a2367efc602553af4c5295b1"
ASRF_PINNED = "9623f1e8d9a1171333a4eeb65d190997b6c44a95"
ASRF_IGNORE = 255


def check_pinned(repo: Path, commit: str) -> str:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"{repo} is at {head}, expected pinned {commit}")
    return head


def _batched(seqs: list[Seq], batch_size: int) -> list[list[Seq]]:
    order = sorted(seqs, key=lambda s: s.labels.size)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


class BoundaryContextHead:
    """M2 (plain: no path, boundary_weight 0), M3 (aux: no path, boundary loss), M4 (full); fixed_gate for the ablation."""

    def __init__(self, model_cfg: dict, boundary_weight: float, name: str) -> None:
        self.cfg = BoundaryContextConfig.from_dict(model_cfg)
        self.boundary_weight, self.name = float(boundary_weight), name

    def build(self) -> nn.Module:
        return BoundaryContextNet(self.cfg)

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        return total_loss(model(batch["x"], batch["mask"]), batch, self.boundary_weight)

    @torch.no_grad()
    def decode(self, model: nn.Module, seqs: list[Seq], device, batch_size: int = 32) -> dict[str, dict[str, np.ndarray]]:
        """Default decoding: per-sample argmax of the mixed logits. Also returns b and the gate for diagnostics."""
        model.eval(); out = {}
        for chunk in _batched(seqs, batch_size):
            b = collate(chunk, device); o = model(b["x"], b["mask"])
            prob = torch.softmax(o["class_logits"], 1).transpose(1, 2).cpu().numpy()
            pre = torch.softmax(o["mixed_logits"], 1).transpose(1, 2).cpu().numpy()   # before pooling; diagnostics only
            lab = o["class_logits"].argmax(1).cpu().numpy(); bp = torch.sigmoid(o["boundary_logits"][:, 0]).cpu().numpy()
            g = o["gate"][:, 0].cpu().numpy(); lp = o["local_logits"].argmax(1).cpu().numpy()
            for i, s in enumerate(chunk):
                n = s.labels.size
                out[s.seq_id] = {"labels": lab[i, :n], "prob": prob[i, :n], "boundary_prob": bp[i, :n], "gate": g[i, :n],
                                 "local_labels": lp[i, :n], "prepool_prob": pre[i, :n]}
        return out


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class MSTCNHead:
    """Official MS-TCN (yabufarha/ms-tcn, pinned): MultiStageModel(4, 10, 64, D, C) with the authors' loss from
    Trainer.train: per stage, cross-entropy plus 0.15 * mean(clamp(MSE of adjacent log-probabilities, 0, 16) * mask)."""

    name = "mstcn"

    def __init__(self, repo: Path, in_dim: int, num_classes: int = 5, num_stages: int = 4, num_layers: int = 10,
                 num_f_maps: int = 64) -> None:
        self.commit = check_pinned(repo, MSTCN_PINNED)
        self.mod = _load_module("mstcn_official_model", repo / "model.py")
        self.args = (num_stages, num_layers, num_f_maps, in_dim, num_classes); self.num_classes = num_classes

    def build(self) -> nn.Module:
        return self.mod.MultiStageModel(*self.args)

    def _mask(self, batch: dict) -> torch.Tensor:
        return batch["mask"].expand(-1, self.num_classes, -1).contiguous()

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        mask = self._mask(batch); preds = model(batch["x"], mask); c = self.num_classes
        ce_sum = mse_sum = 0.0
        for p in preds:  # verbatim from the official Trainer.train
            ce_sum = ce_sum + F.cross_entropy(p.transpose(2, 1).contiguous().view(-1, c), batch["labels"].view(-1), ignore_index=-100)
            mse_sum = mse_sum + 0.15 * torch.mean(torch.clamp(F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1),
                                                                         F.log_softmax(p.detach()[:, :, :-1], dim=1), reduction="none"),
                                                              min=0, max=16) * mask[:, :, 1:])
        return ce_sum + mse_sum, {"ce": float(ce_sum.detach()), "tmse": float(mse_sum.detach())}

    @torch.no_grad()
    def decode(self, model: nn.Module, seqs: list[Seq], device, batch_size: int = 32) -> dict[str, dict[str, np.ndarray]]:
        model.eval(); out = {}
        for chunk in _batched(seqs, batch_size):
            b = collate(chunk, device); last = model(b["x"], self._mask(b))[-1]
            lab = last.argmax(1).cpu().numpy(); prob = torch.softmax(last, 1).transpose(1, 2).cpu().numpy()
            for i, s in enumerate(chunk):
                out[s.seq_id] = {"labels": lab[i, :s.labels.size], "prob": prob[i, :s.labels.size]}
        return out


def asrf_boundary_gt(labels: np.ndarray) -> np.ndarray:
    """ASRF ground truth (utils/generate_boundary_array.py): 1 at the first sample and wherever the label changes."""
    b = np.zeros(labels.size, np.float32)
    if labels.size:
        b[0] = 1.0; b[1:][labels[1:] != labels[:-1]] = 1.0
    return b


def asrf_weights(seqs: list[Seq], num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Official libs/class_weight.py rules on the given training sequences: median-frequency class weights and
    pos_weight = 1 / share of boundary-GT positives."""
    counts = np.bincount(np.concatenate([s.labels for s in seqs]), minlength=num_classes).astype(float)
    freq = counts / counts.sum(); class_weight = np.median(freq) / freq
    gt = np.concatenate([asrf_boundary_gt(s.labels) for s in seqs])
    return torch.tensor(class_weight, dtype=torch.float32), torch.tensor(1.0 / gt.mean())


class ASRFHead:
    """Official ASRF (yiskw713/asrf, pinned): ActionSegmentRefinementFramework with the authors' defaults (64 features,
    10 layers, 4 stages per branch), ActionSegmentationLoss (CE with class weights + Gaussian-similarity TMSE),
    BoundaryRegressionLoss with pos_weight, lambda_b = 0.1, and refinement_with_boundary post-processing at
    boundary_th = 0.5 for decoding. Each sequence is decoded on its own, as in the official evaluation (batch 1)."""

    name = "asrf"

    def __init__(self, repo: Path, in_dim: int, class_weight: torch.Tensor, pos_weight: torch.Tensor, num_classes: int = 5,
                 lambda_b: float = 0.1, boundary_th: float = 0.5, device: str | torch.device = "cpu") -> None:
        self.commit = check_pinned(repo, ASRF_PINNED)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss  # noqa: E402  (official)
        from libs.models.tcn import ActionSegmentRefinementFramework  # noqa: E402  (official)
        from libs.postprocess import PostProcessor  # noqa: E402  (official)
        self._model_cls = ActionSegmentRefinementFramework
        self.in_dim, self.num_classes, self.lambda_b, self.boundary_th = in_dim, num_classes, lambda_b, boundary_th
        self.class_weight, self.pos_weight = class_weight, pos_weight
        self.criterion_cls = ActionSegmentationLoss(ce=True, focal=False, tmse=False, gstmse=True, weight=class_weight.to(device),
                                                    ignore_index=ASRF_IGNORE, ce_weight=1.0, focal_weight=1.0, tmse_weight=0.15,
                                                    gstmse_weight=True)  # train.py passes config.gstmse (True) here
        self.criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight.to(device))
        self.post = PostProcessor("refinement_with_boundary", boundary_th)

    def build(self) -> nn.Module:
        return self._model_cls(in_channel=self.in_dim, n_features=64, n_classes=self.num_classes, n_stages=4, n_layers=10,
                               n_stages_asb=4, n_stages_brb=4)

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch["x"]; t = batch["labels"].masked_fill(batch["labels"] < 0, ASRF_IGNORE)
        b = self._boundary(batch); mask = batch["mask"] > 0
        out_cls, out_bound = model(x)
        sim = x.detach()   # Gaussian-similarity index of the official loss: a fixed weighting, never a gradient path
        cls = sum(self.criterion_cls(o, t, sim) for o in out_cls) / len(out_cls)
        bnd = sum(self.criterion_bound(o, b, mask) for o in out_bound) / len(out_bound)
        return cls + self.lambda_b * bnd, {"cls": float(cls.detach()), "boundary": float(bnd.detach())}

    @staticmethod
    def _boundary(batch: dict) -> torch.Tensor:
        """asrf_boundary_gt on a padded batch -> (B, 1, T); padding is 0 and excluded by the loss mask."""
        lab = batch["labels"]; valid = batch["valid"] > 0
        change = torch.zeros_like(valid); change[:, 1:] = (lab[:, 1:] != lab[:, :-1]) & valid[:, 1:]
        change[:, 0] = valid[:, 0]
        return change.float()[:, None, :]

    @torch.no_grad()
    def decode(self, model: nn.Module, seqs: list[Seq], device, batch_size: int = 1) -> dict[str, dict[str, np.ndarray]]:
        model.eval(); out = {}
        for s in seqs:
            x = torch.from_numpy(s.feat.T.astype(np.float32))[None].to(device)
            cls, bnd = model(x)
            cls = cls.cpu().numpy(); bprob = torch.sigmoid(bnd).cpu().numpy()   # probabilities, so no logit is mistaken for one
            mask = np.ones((1, 1, s.labels.size), bool)
            refined = self.post(cls, boundaries=bprob.copy(), masks=mask)[0]
            prob = torch.softmax(torch.from_numpy(cls[0]), 0).T.numpy()
            out[s.seq_id] = {"labels": refined.astype(int), "frame_labels": cls[0].argmax(0), "prob": prob, "boundary_prob": bprob[0, 0]}
        return out


def head_from_manifest(manifest: dict, repo_root: Path, in_dim: int, device) -> Any:
    """Rebuild the head object of a finished run from its run_manifest.json (decoding and diagnostics only)."""
    name = manifest["head"]
    if name.startswith("boundary_context"):
        return BoundaryContextHead(manifest["model_cfg"], manifest["boundary_weight"], name)
    if name == "mstcn":
        up = manifest["upstream"]
        return MSTCNHead(repo_root / "third_party" / "ms-tcn", in_dim, num_stages=up["num_stages"], num_layers=up["num_layers"],
                         num_f_maps=up["num_f_maps"])
    if name == "asrf":
        up = manifest["upstream"]
        return ASRFHead(repo_root / "third_party" / "asrf", in_dim, torch.tensor(manifest["class_weight"]),
                        torch.tensor(manifest["pos_weight"]), lambda_b=up["lambda_b"], boundary_th=up["boundary_th"], device=device)
    raise KeyError(f"unknown head {name!r}")
