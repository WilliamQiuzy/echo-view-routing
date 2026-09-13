"""B4: train/apply MS-TCN on cached frozen features over constructed training streams."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from echo_routing.compose.stream_assembler import Stream
from echo_routing.temporal.models.mstcn import MultiStageTCN, mstcn_loss


def _batch(streams: Sequence[Stream], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(s.feat.shape[0] for s in streams); D = streams[0].feat.shape[1]
    x = torch.zeros(len(streams), D, T); y = torch.full((len(streams), T), -100, dtype=torch.long); m = torch.zeros(len(streams), 1, T)
    for i, s in enumerate(streams):
        n = s.feat.shape[0]
        x[i, :, :n] = torch.from_numpy(s.feat.T.astype(np.float32)); y[i, :n] = torch.from_numpy(s.labels); m[i, 0, :n] = 1
    return x.to(device), y.to(device), m.to(device)


def train_mstcn(train_streams: Sequence[Stream], val_streams: Sequence[Stream], num_classes: int, cfg: dict,
                out_dir: Path, device: torch.device) -> Path:
    c = cfg["mstcn"]; out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dim = train_streams[0].feat.shape[1]
    model = MultiStageTCN(int(c["stages"]), int(c["layers"]), int(c["channels"]), in_dim, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(cfg.get("seed", 0)); bs = int(c["batch_size"]); best, best_path = -1.0, out_dir / "best.pt"
    log = []
    for epoch in range(int(c["epochs"])):
        model.train(); order = rng.permutation(len(train_streams)); losses = []
        for i in range(0, len(order), bs):
            x, y, m = _batch([train_streams[j] for j in order[i:i + bs]], device)
            loss = mstcn_loss(model(x, m), y, m)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        acc = evaluate_frame_accuracy(model, val_streams, device)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_frame_acc": acc}); print(json.dumps(log[-1]), flush=True)
        torch.save({"state_dict": model.state_dict(), "config": c, "in_dim": in_dim, "num_classes": num_classes, "epoch": epoch}, out_dir / "last.pt")
        if acc > best:
            best = acc; torch.save({"state_dict": model.state_dict(), "config": c, "in_dim": in_dim, "num_classes": num_classes, "epoch": epoch}, best_path)
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    return best_path


def load_mstcn(path: Path, device: torch.device) -> MultiStageTCN:
    p = torch.load(Path(path), map_location=device, weights_only=False); c = p["config"]
    model = MultiStageTCN(int(c["stages"]), int(c["layers"]), int(c["channels"]), p["in_dim"], p["num_classes"]).to(device)
    model.load_state_dict(p["state_dict"]); model.eval()
    return model


@torch.no_grad()
def predict_prob(model: MultiStageTCN, feat: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(feat.T.astype(np.float32))[None].to(device); m = torch.ones(1, 1, feat.shape[0], device=device)
    return torch.softmax(model(x, m)[-1], dim=1)[0].T.cpu().numpy()


@torch.no_grad()
def evaluate_frame_accuracy(model: MultiStageTCN, streams: Sequence[Stream], device) -> float:
    model.eval(); correct = total = 0
    for s in streams:
        pred = predict_prob(model, s.feat, device).argmax(1); correct += int((pred == s.labels).sum()); total += s.labels.size
    return correct / max(total, 1)


_MODEL_CACHE: dict[str, tuple[MultiStageTCN, torch.device]] = {}


def mstcn_decoder(cfg: dict):
    """Decoder closure for the registry: uses cfg["mstcn"]["checkpoint"]; MS-TCN posteriors are used for scoring."""
    ckpt = cfg.get("mstcn", {}).get("checkpoint")
    if not ckpt:
        raise KeyError("b4 requires cfg['mstcn']['checkpoint'] (use --set mstcn.checkpoint=<path>)")
    if ckpt not in _MODEL_CACHE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL_CACHE[ckpt] = (load_mstcn(Path(ckpt), device), device)
    model, device = _MODEL_CACHE[ckpt]

    def _decode(prob, feat, cfg):
        if feat is None:
            raise ValueError("b4 needs frozen features")
        p = predict_prob(model, np.asarray(feat, dtype=np.float32), device)
        return p.argmax(1), p

    return _decode
