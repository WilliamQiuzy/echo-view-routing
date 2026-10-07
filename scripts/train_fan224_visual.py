#!/usr/bin/env python
"""Train one arm (L0, L1, F0, F1) and seed of the fan224 visual adaptation round
(configs/experiments/fan224_visual_adaptation_v1.yaml; plan Task 2).

Model: the E1 image encoder (the frozen export) with a five-class linear head started from the E1 probe. L0 and F0
train the head only: the whole encoder, ln_post and proj included, is frozen and runs in eval mode without autograd.
L1 and F1 also train ViT block 11, ln_post and proj. L reads the legacy E1 input from the source JPEGs (gamma LUT,
short side 256, centre crop 224); F reads the stored fan224_v1 PNGs exactly as they are. Both are normalised with the
existing CLIP mean/std on the GPU.

Data: the protocol's 3,364 training recordings only. One batch plan per seed, saved once and shared by the four arms:
16 recordings per update (class uniform, then recording uniform), 4 spaced 10 Hz samples per recording, orig or
gamma090 per recording. Mean per-frame cross-entropy; AdamW (head 1e-3, encoder 1e-5, weight decay 1e-4); global norm
clip 1.0; 100 linear warm-up updates, then cosine to 0; 1,500 updates; fp32 without TF32; the final update is kept.
Resume state every 250 updates. No validation or test data is read.

--smoke runs the plan's checks on the first 6 updates instead of training: encoder parity (export reference and E1
cache; L only), the PNG contract (F only), the trainable set, same-seed reproducibility, resume (3 + 3 against 6),
unchanged frozen weights, finite non-zero gradients on every trained tensor, and the batch-plan digests.

  python scripts/train_fan224_visual.py --config configs/experiments/fan224_visual_adaptation_v1.yaml --arm F1 --seed 0 [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.features.sampling import spaced_indices  # noqa: E402
from echo_routing.ingest import fan224  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_frame_u8, load_vision_transformer  # noqa: E402

NS = "fan224_visual_adaptation_v1"
ARMS = ("L0", "L1", "F0", "F1")
CODE = ("echo_routing/ingest/fan224.py", "scripts/train_fan224_visual.py")
TRAINED_ENCODER = ("transformer.resblocks.11.", "ln_post.", "proj")
VARIANT = {1.0: "orig", 0.9: "gamma090"}


# ----------------------------------------------------------------------------------------------- inputs and plan
def check_inputs(fv: dict, root: Path) -> dict:
    out = {}
    for key in ("protocol", "visual_export", "visual_reference", "probe", "feature_manifest"):
        spec = fv["inputs"][key]
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {key} {spec['path']}: sha256 {got[:12]} != {spec['sha256'][:12]}")
        out[key] = got
    if fan224.spec_digest() != fv["input_types"]["F"]["transform_digest"]:
        raise SystemExit("echo_routing/ingest/fan224.py does not match the frozen fan224_v1 transform digest")
    f = fv["input_types"]["F"]
    if (got := sha256_file(root / f["train_manifest"])) != f["train_manifest_sha256"]:
        raise SystemExit(f"the fan224 training manifest differs from the frozen export: {got[:12]}")
    out["fan224_train_manifest"] = got
    return out


def train_table(fv: dict, paths) -> dict:
    """The training recordings: id, label (E1 index), 10 Hz frame indices (cache_v2). Training split only."""
    root = paths.repo_root; proto = json.loads((root / fv["inputs"]["protocol"]["path"]).read_text())
    clips = sorted(proto["train"]["clips"])
    idx = read_index(variant_dir(paths.cache_root, fv["inputs"]["cache_key"], "orig")).set_index("video_id")
    man = load_manifest(paths.manifest_path).set_index("video_id"); names = list(fv["classes"])
    if (idx.loc[clips, "split"] != "train").any() or (man.loc[clips, "split"] != "train").any():
        raise SystemExit("a protocol training clip is not in the training split")
    labels = idx.loc[clips, "label_index"].astype(int).to_numpy()
    if [names[k] for k in labels] != man.loc[clips, "family5_label"].tolist():
        raise SystemExit("E1 index labels and manifest labels disagree")
    store = FeatureStore(paths.cache_root, fv["inputs"]["cache_key"])
    frames = [np.asarray(store.get(v, "orig").frame_idx, dtype=np.int64) for v in clips]
    return {"videos": clips, "labels": labels, "frames": frames, "frames_dir": {v: str(man.loc[v, "frames_dir"]) for v in clips}}


def make_plan(fv: dict, table: dict, seed: int) -> dict:
    """Per update and slot: recording, its label, 4 sample positions and frame indices, gamma. RNG call order per slot:
    class, recording, positions, gamma."""
    sp = fv["sampling"]; n_upd, n_rec, k = int(sp["updates"]), int(sp["recordings_per_update"]), int(sp["samples_per_recording"])
    if sp["gamma"] != {"orig": 0.5, "gamma090": 0.5}:
        raise SystemExit("the plan implements gamma orig/gamma090 at 0.5/0.5 only")
    rng = np.random.default_rng([int(sp["data_seed_base"]), int(seed)])
    by_class = [np.flatnonzero(table["labels"] == c) for c in range(len(fv["classes"]))]
    vid = np.zeros((n_upd, n_rec), np.int64); pos = np.zeros((n_upd, n_rec, k), np.int64); frame = np.zeros_like(pos)
    gamma = np.zeros((n_upd, n_rec), np.float64)
    for u in range(n_upd):
        for j in range(n_rec):
            c = int(rng.integers(len(by_class))); v = int(by_class[c][int(rng.integers(len(by_class[c])))])
            n = len(table["frames"][v])
            p = spaced_indices(n, k, rng) if n >= k else np.sort(rng.choice(n, k, replace=True))
            vid[u, j], pos[u, j] = v, p; frame[u, j] = table["frames"][v][p]
            gamma[u, j] = 0.9 if rng.random() < 0.5 else 1.0
    return {"videos": np.array(table["videos"]), "vid": vid, "pos": pos, "frame": frame, "gamma": gamma, "label": table["labels"][vid]}


def update_digests(plan: dict) -> list[str]:
    out = []
    for u in range(plan["vid"].shape[0]):
        h = hashlib.sha256()
        for j in range(plan["vid"].shape[1]):
            h.update(f"{plan['videos'][plan['vid'][u, j]]}|{','.join(map(str, plan['frame'][u, j]))}|{plan['gamma'][u, j]:.2f};".encode())
        out.append(h.hexdigest()[:16])
    return out


def save_or_check_plan(fv: dict, table: dict, seed: int, root: Path) -> tuple[Path, dict]:
    path = root / "runs" / NS / "plans" / f"plan_s{seed}.npz"; plan = make_plan(fv, table, seed)
    if path.is_file():
        with np.load(path) as z:
            old = {k: z[k] for k in z.files}
        if set(old) != set(plan) or any(not np.array_equal(old[k], plan[k]) for k in plan):
            raise SystemExit(f"{path} exists and differs from a fresh generation")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz"); np.savez(tmp, **plan); tmp.replace(path)
    return path, plan


# ----------------------------------------------------------------------------------------------- model
def load_probe_state(fv: dict, root: Path) -> dict:
    blob = torch.load(root / fv["inputs"]["probe"]["path"], map_location="cpu", weights_only=True)
    if list(blob["classes"]) != list(fv["classes"]) or int(blob["in_dim"]) != 512:
        raise SystemExit("the E1 probe's classes or input size differ from the round's")
    return {k: v.clone() for k, v in blob["state_dict"].items()}


def build(fv: dict, paths, arm: str, device) -> tuple[VisualEncoder, torch.nn.Linear]:
    root = paths.repo_root
    vt, _ = load_vision_transformer(root / "third_party" / "EchoViewCLIP", root / fv["inputs"]["visual_export"]["path"])
    enc = VisualEncoder(vt, trainable_blocks=1, chunk=128)
    if arm in ("L0", "F0"):                       # head only: the whole encoder, ln_post and proj included
        for p in enc.parameters():
            p.requires_grad_(False)
    head = torch.nn.Linear(512, len(fv["classes"])); head.load_state_dict(load_probe_state(fv, root))
    names = [n for n, p in enc.vt.named_parameters() if p.requires_grad]
    if arm in ("L0", "F0") and names:
        raise SystemExit(f"{arm}: encoder tensors still trainable: {names[:3]}")
    if arm in ("L1", "F1") and (not names or not all(n.startswith(TRAINED_ENCODER) for n in names)
                                or not any(n.startswith("transformer.resblocks.11.") for n in names)
                                or not any(n.startswith("ln_post.") for n in names) or "proj" not in names):
        raise SystemExit(f"{arm}: trainable encoder set is not block 11, ln_post and proj: {names}")
    return enc.to(device), head.to(device)


def init_state(enc: VisualEncoder, head: torch.nn.Linear) -> dict:
    return {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "encoder_trainable_part": {n: p.detach().cpu().clone() for n, p in enc.vt.named_parameters() if n.startswith(TRAINED_ENCODER)}}


def save_or_check_init(state: dict, seed: int, root: Path) -> tuple[Path, str]:
    path = root / "runs" / NS / "inits" / f"init_s{seed}.pt"
    if path.is_file():
        old = torch.load(path, map_location="cpu", weights_only=True)
        same = all(set(old[g]) == set(state[g]) and all(torch.equal(old[g][k], state[g][k]) for k in state[g]) for g in state)
        if not same:
            raise SystemExit(f"{path} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)
    return path, sha256_file(path)


def forward(enc: VisualEncoder, head: torch.nn.Linear, u8: torch.Tensor, arm: str) -> torch.Tensor:
    if arm in ("L0", "F0"):
        enc.eval()
        with torch.no_grad():
            z = enc(u8)
    else:
        enc.train(); z = enc(u8)
    head.train()
    return head(z)


# ----------------------------------------------------------------------------------------------- data
class UpdateFrames(Dataset):
    """Item u = the 64 frames of update u (slot-major, 4 samples per slot) as uint8 (64, 224, 224, 3) and their labels."""

    def __init__(self, plan: dict, kind: str, frame_paths, fan_root: Path, updates: list[int]) -> None:
        if kind not in ("legacy", "fan224_v1"):
            raise ValueError(kind)
        self.plan, self.kind, self.frame_paths, self.fan_root, self.updates = plan, kind, frame_paths, Path(fan_root), list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def frame(self, video: str, f: int, gamma: float) -> np.ndarray:
        if self.kind == "legacy":
            paths = self.frame_paths(video)
            return load_frame_u8(paths[min(int(f), len(paths) - 1)], gamma)
        return fan224.read_fan224_png(self.fan_root / VARIANT[round(gamma, 2)] / video / f"{int(f):06d}.png")

    def __getitem__(self, i: int):
        u = self.updates[i]; p = self.plan; frames, labels = [], []
        for j in range(p["vid"].shape[1]):
            v = str(p["videos"][p["vid"][u, j]])
            for f in p["frame"][u, j]:
                frames.append(self.frame(v, int(f), float(p["gamma"][u, j]))); labels.append(int(p["label"][u, j]))
        return torch.from_numpy(np.stack(frames)), torch.tensor(labels), u


def loader(plan: dict, kind: str, frame_paths, fan_root: Path, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(UpdateFrames(plan, kind, frame_paths, fan_root, updates), batch_size=None, num_workers=workers,
                      prefetch_factor=2 if workers else None, pin_memory=True)


# ----------------------------------------------------------------------------------------------- training
def make_optimiser(fv: dict, enc: VisualEncoder, head: torch.nn.Linear):
    o = fv["optimiser"]; enc_params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": float(o["lr_head"])}]
    if enc_params:
        groups.append({"params": enc_params, "lr": float(o["lr_encoder"])})
    opt = torch.optim.AdamW(groups, weight_decay=float(o["weight_decay"])); total = int(fv["sampling"]["updates"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, int(o["warmup_updates"]), total))
    return opt, sched, [p for g in opt.param_groups for p in g["params"]]


def step(enc, head, batch, arm: str, opt, sched, params, clip: float, device) -> dict:
    u8, y, _ = batch
    logits = forward(enc, head, u8.to(device, non_blocking=True), arm); y = y.to(device)
    loss = F.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, clip)); opt.step(); sched.step()
    return {"loss": float(loss.detach()), "acc": float((logits.argmax(1) == y).float().mean()), "grad_norm": gn}


def trained_state(enc: VisualEncoder, head: torch.nn.Linear) -> dict:
    return {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "encoder_trainable": {n: p.detach().cpu().clone() for n, p in enc.vt.named_parameters() if p.requires_grad}}


def save_resume(path: Path, enc, head, opt, sched, updates: int, ident: str, log: list, pixels: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": updates, **trained_state(enc, head), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "log": log, "pixels": pixels}, tmp)
    tmp.replace(path)


def load_resume(path: Path, enc, head, opt, sched, ident: str) -> tuple[int, list, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    with torch.no_grad():
        named = dict(enc.vt.named_parameters())
        for n, v in st["encoder_trainable"].items():
            named[n].copy_(v.to(named[n].device))
    head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["updates"]), list(st["log"]), list(st["pixels"])


def identity(cfg: dict, arm: str, seed: int, inputs: dict, plan_sha: str, init_sha: str, root: Path) -> str:
    fv = cfg["fan224"]; kind = fv["input_types"][fv["arms"][arm]["input"]]["kind"]
    payload = {"experiment": cfg.get("experiment"), "arm": arm, "seed": int(seed), "config": fv, "inputs": inputs, "plan_sha256": plan_sha,
               "init_sha256": init_sha, "input_kind": kind, "transform_digest": fv["input_types"]["F"]["transform_digest"] if kind == "fan224_v1" else None,
               "code": {c: sha256_file(root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def encoder_digests(enc: VisualEncoder) -> dict:
    named = list(enc.vt.named_parameters())
    return {"frozen": params_digest([(n, p) for n, p in named if not n.startswith(TRAINED_ENCODER)]),
            "last_block_part": params_digest([(n, p) for n, p in named if n.startswith(TRAINED_ENCODER)])}


# ----------------------------------------------------------------------------------------------- smoke
@torch.no_grad()
def parity(enc: VisualEncoder, fv: dict, paths, table: dict, frame_paths, device) -> dict:
    enc.eval(); root = paths.repo_root; ref = json.loads((root / fv["inputs"]["visual_reference"]["path"]).read_text())
    u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]])).to(device)
    d_ref = float((enc(u8).cpu() - torch.tensor(ref["reference_embeddings"])).abs().max())
    v = table["videos"][0]
    with np.load(paths.cache_root / "features" / fv["inputs"]["cache_key"] / "orig" / f"{v}.npz") as z:
        feat, fidx = z["feat"].astype(np.float32), z["frame_idx"]
    fp = frame_paths(v); n = min(len(fidx), 32)
    u8 = torch.from_numpy(np.stack([load_frame_u8(fp[min(int(f), len(fp) - 1)], 1.0) for f in fidx[:n]])).to(device)
    d_cache = float(np.abs(enc(u8).cpu().numpy() - feat[:n]).max())
    tol = fv["smoke"]["parity"]
    if d_ref > float(tol["reference_fp32"]) or d_cache > float(tol["cache_fp16"]):
        raise SystemExit(f"encoder parity failed: reference {d_ref:.2e}, cache {d_cache:.2e}")
    return {"reference_max_abs": d_ref, "cache_fp16_max_abs": d_cache, "cache_frames": n}


def png_contract(plan: dict, frame_paths, fan_root: Path, updates: int = 1) -> int:
    n = 0
    for u in range(updates):
        for j in range(plan["vid"].shape[1]):
            v = str(plan["videos"][plan["vid"][u, j]]); g = float(plan["gamma"][u, j]); paths = frame_paths(v)
            for f in plan["frame"][u, j]:
                stored = fan224.read_fan224_png(fan_root / VARIANT[round(g, 2)] / v / f"{int(f):06d}.png")
                with Image.open(paths[min(int(f), len(paths) - 1)]) as im:
                    if not np.array_equal(stored, fan224.fan224(im, g)):
                        raise SystemExit(f"stored fan224 frame differs from the transform: {v} {f} gamma {g}")
                n += 1
    return n


def run_updates(fv, paths, arm, device, plan, kind, frame_paths, fan_root, updates, workers, state=None):
    enc, head = build(fv, paths, arm, device); opt, sched, params = make_optimiser(fv, enc, head)
    if state is not None:   # as load_resume: weights, optimiser, scheduler and RNG after the model is built
        with torch.no_grad():
            named = dict(enc.vt.named_parameters())
            for n, v in state["encoder_trainable"].items():
                named[n].copy_(v.to(device))
        head.load_state_dict(state["head"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng_cpu"]); torch.cuda.set_rng_state_all(state["rng_cuda"])
    losses, grads = [], None
    for batch in loader(plan, kind, frame_paths, fan_root, updates, workers):
        u8, y, _ = batch
        logits = forward(enc, head, u8.to(device), arm); loss = F.cross_entropy(logits, y.to(device))
        opt.zero_grad(set_to_none=True); loss.backward()
        if grads is None:
            grads = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in [*enc.vt.named_parameters(), *(("head." + k, v) for k, v in head.named_parameters())] if p.requires_grad}
        torch.nn.utils.clip_grad_norm_(params, float(fv["optimiser"]["grad_clip"])); opt.step(); sched.step(); losses.append(float(loss.detach()))
    return enc, head, opt, sched, losses, grads


def smoke(cfg, fv, paths, arm, seed, device, plan, kind, frame_paths, fan_root, table, workers) -> dict:
    sm = fv["smoke"]; k = int(sm["updates"]); rep = {"arm": arm, "seed": seed, "input": kind}
    enc0, head0 = build(fv, paths, arm, device); start = encoder_digests(enc0)
    if kind == "legacy":
        rep["parity"] = parity(enc0, fv, paths, table, frame_paths, device)
    else:
        rep["png_contract_frames"] = png_contract(plan, frame_paths, fan_root)
    del enc0, head0
    torch.manual_seed(0)
    a = run_updates(fv, paths, arm, device, plan, kind, frame_paths, fan_root, list(range(k)), workers)
    torch.manual_seed(0)
    b = run_updates(fv, paths, arm, device, plan, kind, frame_paths, fan_root, list(range(k)), workers)
    sa, sb = trained_state(a[0], a[1]), trained_state(b[0], b[1])
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[4], b[4])),
                              "params_max_abs": max(float((sa[g][n] - sb[g][n]).abs().max()) for g in sa for n in sa[g])}
    torch.manual_seed(0)    # the interrupted run starts exactly as the uninterrupted one
    c = run_updates(fv, paths, arm, device, plan, kind, frame_paths, fan_root, list(range(k // 2)), workers)
    st = {**trained_state(c[0], c[1]), "opt": c[2].state_dict(), "sched": c[3].state_dict(),
          "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
    d = run_updates(fv, paths, arm, device, plan, kind, frame_paths, fan_root, list(range(k // 2, k)), workers, state=st)
    sd = trained_state(d[0], d[1])
    rep["resume_params_max_abs"] = max(float((sa[g][n] - sd[g][n]).abs().max()) for g in sa for n in sa[g])
    end = encoder_digests(a[0])
    rep["frozen_unchanged"] = end["frozen"] == start["frozen"]
    rep["last_block_part_changed"] = end["last_block_part"] != start["last_block_part"]
    rep["first_backward_grad_max"] = a[5]
    rep["losses"] = a[4]; rep["update_digests"] = update_digests({**plan, "vid": plan["vid"][:k]})
    zero = [n for n, g in a[5].items() if g is None or not np.isfinite(g) or g == 0.0]
    ok_enc = (not rep["last_block_part_changed"]) if arm in ("L0", "F0") else rep["last_block_part_changed"]
    rep["ok"] = bool(rep["frozen_unchanged"] and ok_enc and not zero and rep["reproducibility"]["loss_max_abs"] <= float(sm["reproducibility"]["loss_abs"])
                     and rep["reproducibility"]["params_max_abs"] <= float(sm["reproducibility"]["params_abs"])
                     and rep["resume_params_max_abs"] <= float(sm["resume"]["params_abs"]))
    rep["zero_or_missing_gradients"] = zero
    return rep


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=ARMS, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fv = cfg["fan224"]; paths = load_paths(); root = paths.repo_root
    if a.seed not in [int(s) for s in fv["seeds"]]:
        raise SystemExit(f"seed {a.seed} is not one of the round's seeds")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    inputs = check_inputs(fv, root); table = train_table(fv, paths)
    plan_path, plan = save_or_check_plan(fv, table, a.seed, root); plan_sha = sha256_file(plan_path)
    enc0, head0 = build(fv, paths, a.arm, torch.device("cpu")); init_path, init_sha = save_or_check_init(init_state(enc0, head0), a.seed, root); del enc0, head0
    kind = fv["input_types"][fv["arms"][a.arm]["input"]]["kind"]; fan_root = root / fv["input_types"]["F"]["frames_root"]
    frame_paths = FramePaths(table["frames_dir"]); ident = identity(cfg, a.arm, a.seed, inputs, plan_sha, init_sha, root)
    key = f"{a.arm}_s{a.seed}"
    if a.smoke:
        rep = smoke(cfg, fv, paths, a.arm, a.seed, device, plan, kind, frame_paths, fan_root, table, a.workers)
        out = root / "runs" / NS / "smoke" / f"{key}.json"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**rep, "identity": ident, "plan_sha256": plan_sha, "init_sha256": init_sha}, indent=1, default=str))
        print(json.dumps({k: v for k, v in rep.items() if k not in ("first_backward_grad_max",)}, indent=1, default=str))
        if not rep["ok"]:
            raise SystemExit("smoke checks failed")
        return
    run_dir = root / "runs" / NS / "train" / key; ckpt = root / "checkpoints" / NS / key
    run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    enc, head = build(fv, paths, a.arm, device); start = encoder_digests(enc)
    opt, sched, params = make_optimiser(fv, enc, head); total = int(fv["sampling"]["updates"]); every = int(fv["resume_every"])
    done, log, pixels = 0, [], []
    if (ckpt / "resume.pt").is_file():
        done, log, pixels = load_resume(ckpt / "resume.pt", enc, head, opt, sched, ident); print(f"{key}: resumed at update {done}", flush=True)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats(device); acc = {"loss": [], "acc": [], "grad_norm": []}
    for batch in loader(plan, kind, frame_paths, fan_root, list(range(done, total)), a.workers):
        pixels.append(hashlib.sha256(batch[0].numpy().tobytes()).hexdigest()[:16])
        rec = step(enc, head, batch, a.arm, opt, sched, params, float(fv["optimiser"]["grad_clip"]), device); done += 1
        for m in acc:
            acc[m].append(rec[m])
        if done % 50 == 0 or done == total:
            lrs = sched.get_last_lr()
            entry = {"updates": done, **{m: float(np.mean(v)) for m, v in acc.items()}, "grad_norm_max": float(np.max(acc["grad_norm"])),
                     "lr_head": lrs[0], "lr_encoder": lrs[1] if len(lrs) > 1 else None, "seconds": time.time() - t0,
                     "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(entry); acc = {m: [] for m in acc}; print(json.dumps(entry), flush=True)
        if done % every == 0 and done < total:
            save_resume(ckpt / "resume.pt", enc, head, opt, sched, done, ident, log, pixels)
    if done != total or len(pixels) != total:
        raise SystemExit(f"{key}: {done} updates and {len(pixels)} input digests, expected {total}")
    end = encoder_digests(enc)
    frozen_ok = end["frozen"] == start["frozen"] and ((end["last_block_part"] == start["last_block_part"]) if a.arm in ("L0", "F0")
                                                     else (end["last_block_part"] != start["last_block_part"]))
    final = ckpt / "final.pt"
    torch.save({"identity": ident, "arm": a.arm, "seed": a.seed, "updates": done, **trained_state(enc, head)}, final)
    man = {"identity": ident, "arm": a.arm, "seed": a.seed, "input_kind": kind, "updates": done, "final_checkpoint": str(final),
           "final_sha256": sha256_file(final), "plan": str(plan_path), "plan_sha256": plan_sha, "init": str(init_path), "init_sha256": init_sha,
           "inputs": inputs, "transform_digest": fv["input_types"]["F"]["transform_digest"] if kind == "fan224_v1" else None,
           "update_digests_sha256": hashlib.sha256("".join(update_digests(plan)).encode()).hexdigest(),
           "pixel_digests_sha256": hashlib.sha256("".join(pixels).encode()).hexdigest(), "pixel_digests": pixels,
           "frozen_unchanged": bool(frozen_ok), "encoder_digests": {"start": start, "end": end}, "log": log}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    if not frozen_ok:
        raise SystemExit(f"{key}: the encoder changed outside the trained part (or the trained part did not change)")
    (ckpt / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"run": key, "updates": done, "minutes": round((time.time() - t0) / 60, 1), "final_sha256": man["final_sha256"][:12]}), flush=True)


if __name__ == "__main__":
    main()
