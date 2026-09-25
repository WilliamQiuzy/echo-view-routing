#!/usr/bin/env python
"""ViT-Router: train the stream-level temporal Transformer on frozen ViT-S frame features, export per-stream posteriors.

Training data = the same class-balanced multi-fragment TRAIN bank used for the official MS-TCN and ASFormer
(compose.banks.multi_bank: 1,500 streams, 4-8 fragments, gamma edits on ~50% of fragments), built from the B0 index
(--bank-hash) so the recipes are identical; features come from the ViT cache (--feat-hash). As in the official
MS-TCN/ASFormer code, the final epoch is used (no checkpoint selection on validation). Evaluation streams come from
plans.json, identical to every other method.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank, render_bank  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.vit.infer import assemble  # noqa: E402
from echo_routing.vit.optim import warmup_cosine  # noqa: E402
from echo_routing.vit.router import ViTRouter, router_loss  # noqa: E402


def main() -> None:
    ap = base_parser("Train ViT-Router on cached ViT features")
    ap.add_argument("--feat-hash", required=True); ap.add_argument("--bank-hash", default="79f4a41af6e3")
    ap.add_argument("--plans", default="data/window_plans/79f4a41af6e3/plans.json")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    seed = int(cfg["seed"]); torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    cache: dict = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.feat_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    idx = read_index(variant_dir(paths.cache_root, args.bank_hash, "orig"))
    label_of = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    bank = multi_bank(idx, "train", cfg)
    train = [(torch.from_numpy(s.feat).float(), torch.from_numpy(s.labels).long()) for s in render_bank(bank, loader)]
    plans = json.loads((paths.repo_root / args.plans).read_text())["streams"]
    val = [(torch.from_numpy(assemble(p["refs"], loader, "feat")).float(), np.array([label_of[r[0]] for r in p["refs"]]))
           for p in plans.values() if p["bank"] in ("pairs_validation", "multi_validation")]
    in_dim = int(train[0][0].shape[1]); n_classes = 5
    net = ViTRouter(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train); warm = len(train)
    run_dir = new_run_dir(cfg, "vit_router"); step = 0
    print(json.dumps({"train_streams": len(train), "val_streams": len(val), "in_dim": in_dim, "feat_hash": args.feat_hash,
                      "params": sum(p.numel() for p in net.parameters())}), flush=True)
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        net.train(); t0 = time.time(); losses = []
        for i in rng.permutation(len(train)):
            x, y = train[i]
            for g in opt.param_groups:
                g["lr"] = args.lr * warmup_cosine(step, total, warm)
            loss = router_loss(net(x[None].to(device)), y[None].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(loss.item()); step += 1
        msg = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            net.eval(); correct = total_n = 0
            with torch.no_grad():   # monitoring only: the final epoch is exported regardless
                for x, truth in val:
                    pred = net(x[None].to(device))[-1].argmax(-1)[0].cpu().numpy()
                    correct += int((pred == truth).sum()); total_n += truth.size
            msg["val_frame_acc"] = correct / max(total_n, 1)
        print(json.dumps(msg), flush=True)
    ckpt_dir = paths.checkpoints_root / run_dir.name; ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": in_dim, "n_classes": n_classes, "feat_hash": args.feat_hash,
                "epochs": args.epochs, "lr": args.lr, "seed": seed}, ckpt_dir / "router_final.pt")
    out = paths.cache_root / "vit_streams" / (args.tag or f"vit_router_{args.feat_hash}_s{seed}"); out.mkdir(parents=True, exist_ok=True)
    net.eval()
    with torch.no_grad():
        for sid, p in plans.items():
            x = torch.from_numpy(assemble(p["refs"], loader, "feat")).float()[None].to(device)
            prob = torch.softmax(net(x)[-1].float(), -1)[0].cpu().numpy()
            np.savez_compressed(out / f"{sid}.npz", prob=prob.astype(np.float32))
    (run_dir / "pred_dir.txt").write_text(str(out))
    print(f"pred_dir={out}\nrun_dir={run_dir}\nckpt={ckpt_dir / 'router_final.pt'}")


if __name__ == "__main__":
    main()
