#!/usr/bin/env python
"""Train one arm and seed of the fold-output correction round (plan Task 2; configs/experiments/fold_output_correction_v1.yaml).

T-in and T-out: the official ASRF (pinned commit; model, losses and post-processing imported unchanged through
ASRFHead) on five-class probabilities [T, 5]. P-out: a per-frame log-probability linear calibrator,
logits_t = W log(clamp(p_t, 1e-6)) + b, from W = I and b = 0, trained with class-weighted cross entropy only.

Each update reads 4 of the 1,500 library streams and 4 native windows of the seed's saved plan, materialised for the arm
(T-in: the model that trained on the fold; T-out and P-out: the model that did not). 50 epochs x 375 updates; AdamW
lr 3e-4, weight decay 1e-4, clip 1.0; 2 warm-up epochs then cosine to 0; fp32, TF32 off. Nothing is evaluated during
training and the final weights are kept. T-in and T-out of a seed load one saved initial state; the RNG is seeded right
before the first update and restored on resume. Resume state after every epoch.

  --smoke   same-state reproducibility, resume (3 + 3 updates against 6), padding labels without effect on the loss,
            finite non-zero gradients on the real five-channel input, the plan and initial state, P-out's identity start
            and per-frame property, and timing / memory for the budget.
  --queue   the nine runs in the configured order (seed 0: T-in, T-out, P-out; seed 1; seed 2); finished runs are skipped.

  python scripts/train_fold_output_correction.py --config configs/experiments/fold_output_correction_v1.yaml --arm T-out --seed 0 [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
import scripts.prepare_fold_output_correction as pf  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_data import PAD_LABEL, collate  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor, seed_everything  # noqa: E402

NS = pf.NS
CODE = ("scripts/train_fold_output_correction.py", "scripts/prepare_fold_output_correction.py", "echo_routing/temporal/dev_heads.py",
        "echo_routing/temporal/dev_data.py", "echo_routing/temporal/dev_train.py")
ASRF_BUILD = {"in_channel": 5, "n_classes": 5, "n_features": 64, "n_layers": 10, "n_stages": 4, "n_stages_asb": 4, "n_stages_brb": 4}


# ----------------------------------------------------------------------------------------------- P-out
class LinearCalibrator(nn.Module):
    """logits_t = W log(clamp(p_t, min)) + b over the class axis of x (B, 5, T); no neighbouring frame is read."""

    def __init__(self, n_classes: int = 5, clamp_min: float = 1e-6) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_classes)); self.b = nn.Parameter(torch.zeros(n_classes)); self.clamp_min = float(clamp_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjt->bit", self.W, torch.log(torch.clamp(x, min=self.clamp_min))) + self.b[None, :, None]


class CalibratorHead:
    """P-out: class-weighted cross entropy only (the valid mask through the padding label); decoding is the report's."""

    name = "p_out"

    def __init__(self, class_weight: torch.Tensor, clamp_min: float, n_classes: int = 5) -> None:
        self.class_weight, self.clamp_min, self.n_classes = class_weight, float(clamp_min), n_classes

    def build(self) -> nn.Module:
        return LinearCalibrator(self.n_classes, self.clamp_min)

    def loss(self, model: nn.Module, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"].masked_fill(batch["valid"] == 0, PAD_LABEL)          # the valid mask, whatever the padding label
        loss = F.cross_entropy(model(batch["x"]), labels, weight=self.class_weight.to(batch["x"].device), ignore_index=PAD_LABEL)
        return loss, {"ce": float(loss.detach())}

    @torch.no_grad()
    def probs(self, model: nn.Module, prob: np.ndarray, device) -> np.ndarray:
        """q (T, 5) of one sequence's probabilities (T, 5)."""
        x = torch.from_numpy(np.asarray(prob, np.float32).T)[None].to(device)
        return torch.softmax(model(x), 1)[0].T.cpu().numpy()


# ----------------------------------------------------------------------------------------------- run pieces
def make_head(fo: dict, root: Path, arm: str, plan: dict, device):
    cw = torch.tensor(plan["class_weight"], dtype=torch.float32)
    if arm == "P-out":
        return CalibratorHead(cw, float(fo["p_out"]["clamp_min"]))
    from echo_routing.temporal.dev_heads import ASRFHead
    a = fo["asrf"]
    if {k: int(v) for k, v in a["model"].items()} != ASRF_BUILD:
        raise SystemExit(f"the config's ASRF architecture {a['model']} differs from what ASRFHead.build constructs {ASRF_BUILD}")
    head = ASRFHead(root / a["repo"], int(a["model"]["in_channel"]), cw, torch.tensor(plan["pos_weight"]), num_classes=int(a["model"]["n_classes"]),
                    lambda_b=float(a["lambda_b"]), boundary_th=float(a["boundary_th"]), device=device)
    if head.commit != a["commit"]:
        raise SystemExit(f"ASRF commit {head.commit} != {a['commit']}")
    return head


def state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_or_check_init(head, arm: str, seed: int, root: Path) -> tuple[Path, str]:
    """One ASRF initial state per seed, shared by T-in and T-out; P-out's identity state is saved the same way."""
    name = "p_out_identity.pt" if arm == "P-out" else f"asrf_init_s{seed}.pt"; path = root / "runs" / NS / "inits" / name
    seed_everything(seed); state = {k: v.detach().cpu().clone() for k, v in head.build().state_dict().items()}
    if path.is_file():
        old = torch.load(path, map_location="cpu", weights_only=True)
        if set(old) != set(state) or any(not torch.equal(old[k], state[k]) for k in state):
            raise SystemExit(f"{path} exists with other content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(state, path)
    return path, sha256_file(path)


def new_model(head, init_path: Path, device) -> nn.Module:
    model = head.build().to(device); model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True)); return model


def make_optimiser(model: nn.Module, tr: dict):
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    total = int(tr["epochs"]) * int(tr["updates_per_epoch"]); warm = int(tr["warmup_epochs"]) * int(tr["updates_per_epoch"])
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))


def update(head, model: nn.Module, seqs: list, opt, sched, clip: float, device) -> dict:
    model.train(); batch = collate(seqs, device); loss, parts = head.loss(model, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {parts}")
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    opt.step(); sched.step()
    return {"total": float(loss.detach()), **parts, "grad_norm": gn}


def optimizer_bound(model: nn.Module, opt) -> bool:
    """Every optimizer parameter object belongs to the model that runs forward."""
    ids = {id(p) for p in model.parameters()}
    return all(id(p) in ids for g in opt.param_groups for p in g["params"])


def save_resume(path: Path, model, opt, sched, epoch: int, ident: str, records: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "records": records}, tmp)
    tmp.replace(path)


def load_resume(path: Path, model, opt, sched, ident: str) -> tuple[int, list]:
    """Loads into the existing model, optimizer and scheduler, so the optimizer stays bound to the model that runs."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng_cpu"])
    if torch.cuda.is_available() and st["rng_cuda"]:
        torch.cuda.set_rng_state_all(st["rng_cuda"])
    if not optimizer_bound(model, opt):
        raise SystemExit("the optimizer is not bound to the resumed model")
    return int(st["epoch"]) + 1, list(st["records"])


def identity(cfg: dict, arm: str, seed: int, input_keys: dict, input_content: dict, plan_sha: str, init_sha: str, root: Path) -> str:
    payload = {"experiment": cfg.get("experiment"), "arm": arm, "seed": int(seed), "config": cfg["fold_output_correction"], "input_keys": input_keys,
               "input_content": input_content, "plan_sha256": plan_sha, "init_sha256": init_sha, "code": {c: sha256_file(root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def finished_run(manifest: Path, ident: str):
    """True: finished with this identity and an intact final checkpoint; False: finished otherwise; None: not finished."""
    if not manifest.is_file():
        return None
    man = json.loads(manifest.read_text())
    return bool(man.get("identity") == ident and Path(man["final_checkpoint"]).is_file() and sha256_file(Path(man["final_checkpoint"])) == man["final_sha256"])


class Setup:
    """Everything one arm and seed needs: the context, plan, head, initial state and identity."""

    def __init__(self, cfg: dict, arm: str, seed: int, paths, device) -> None:
        fo = cfg["fold_output_correction"]; root = paths.repo_root; self.tr = fo["training"]; self.arm, self.seed = arm, seed
        pf.check_inputs(fo, root); self.ctx = pf.load_context(cfg, fo, root); self.n_samples = pf.n_samples_of(self.ctx, paths.cache_root)
        self.arms = pf.ArmInputs(paths.cache_root, self.ctx.keys, self.ctx.train_sessions)
        plan = pf.make_plan(self.ctx, self.tr, self.n_samples, seed); self.plan_path, self.plan_sha = pf.save_or_check_plan(plan, root); self.plan = plan
        self.head = make_head(fo, root, arm, plan, device); self.init_path, self.init_sha = save_or_check_init(self.head, arm, seed, root)
        pf.caches(self.ctx, fo, paths.cache_root, set(self.ctx.proto["train"]["clips"]))   # T-in / T-out coverage and frame alignment, again
        self.input_keys = {str(k): pf.arm_cache_key(arm, k, self.ctx.keys) for k in range(pf.N_FOLDS)}
        self.input_content = {k: pf.cache_digest(paths.cache_root, key, [v for v, _ in self.ctx.clips if self.ctx.fold_of_clip[v] == int(k)])
                              for k, key in self.input_keys.items()}
        self.ident = identity(cfg, arm, seed, self.input_keys, self.input_content, self.plan_sha, self.init_sha, root)

    def batches(self, epoch: int) -> list[list]:
        items = pf.plan_of(self.ctx, self.tr, self.n_samples, self.seed, epoch)
        if [pf.batch_digest(x) for x in items] != self.plan["epochs"][epoch]:
            raise SystemExit(f"epoch {epoch}: batches differ from the saved plan")
        return [[self.arms.seq(self.arm, it) for it in x] for x in items]


# ----------------------------------------------------------------------------------------------- smoke
def smoke(s: Setup, device) -> dict:
    tr = s.tr; clip = float(tr["grad_clip"]); b0 = s.batches(0)[:6]; rep = {}

    def run(batches, state=None):
        model = new_model(s.head, s.init_path, device); opt, sched = make_optimiser(model, tr)
        if state:
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); torch.set_rng_state(state["rng_cpu"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["rng_cuda"])
        else:
            seed_everything(s.seed)
        losses = [update(s.head, model, b, opt, sched, clip, device)["total"] for b in batches]
        return model, opt, sched, losses
    a = run(b0); b = run(b0); pa, pb = a[0].state_dict(), b[0].state_dict()
    rep["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(a[3], b[3])), "params_max_abs": max(float((pa[k].float() - pb[k].float()).abs().max()) for k in pa)}
    c = run(b0[:3]); st = {"model": c[0].state_dict(), "opt": c[1].state_dict(), "sched": c[2].state_dict(), "rng_cpu": torch.get_rng_state(),
                           "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    d = run(b0[3:], state=st); pd_ = d[0].state_dict()
    rep["resume_params_max_abs"] = max(float((pa[k].float() - pd_[k].float()).abs().max()) for k in pa); rep["resume_optimizer_bound"] = optimizer_bound(d[0], d[1])
    model = new_model(s.head, s.init_path, device); model.train(); batch = collate(b0[0], device)
    rep["input_shape"] = list(batch["x"].shape); rep["input_rows_sum_to_one"] = bool(torch.allclose(batch["x"].sum(1)[batch["valid"] > 0], torch.ones(1, device=device), atol=1e-3))
    torch.manual_seed(1); l1, _ = s.head.loss(model, batch); alt = dict(batch); lab = batch["labels"].clone(); lab[batch["valid"] == 0] = -7; alt["labels"] = lab
    torch.manual_seed(1); l2, _ = s.head.loss(model, alt)
    torch.manual_seed(1); l3, _ = s.head.loss(model, batch)                     # the same input twice: the device's numerical noise
    rep["padding_label_loss_diff"] = abs(float(l1) - float(l2)); rep["same_input_loss_diff"] = abs(float(l1) - float(l3))
    rep["padded_positions"] = int((batch["valid"] == 0).sum()); rep["loss_first"] = float(l1)
    model.zero_grad(); l1.backward()
    grads = {n: (float(p.grad.abs().max()) if p.grad is not None else None) for n, p in model.named_parameters()}
    rep["zero_or_missing_gradients"] = [n for n, g in grads.items() if g is None or g == 0.0 or not np.isfinite(g)]; rep["parameters"] = sum(p.numel() for p in model.parameters())
    if s.arm == "P-out":
        rep["p_out"] = p_out_checks(s.head, s.init_path, b0[0][0].feat, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); bs = s.batches(1); rep["seconds_per_epoch_of_batches"] = time.time() - t0     # one whole epoch, caches warm
    model2 = new_model(s.head, s.init_path, device); opt, sched = make_optimiser(model2, tr); bs = bs[:50]; t1 = time.time()
    for x in bs:
        update(s.head, model2, x, opt, sched, clip, device)
    rep["seconds_per_update"] = (time.time() - t1) / len(bs)
    rep["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None
    rep["estimated_minutes_per_run"] = (rep["seconds_per_update"] * int(tr["updates_per_epoch"]) + rep["seconds_per_epoch_of_batches"]) * int(tr["epochs"]) / 60
    p_ok = s.arm != "P-out" or (rep["p_out"]["identity_max_abs"] <= 1e-6 and rep["p_out"]["neighbour_change_max_abs"] == 0.0)
    rep["ok"] = bool(rep["reproducibility"]["loss_max_abs"] <= 1e-6 and rep["reproducibility"]["params_max_abs"] <= 1e-6 and rep["resume_params_max_abs"] <= 1e-6
                     and rep["resume_optimizer_bound"] and rep["padding_label_loss_diff"] <= 1e-6 and rep["padded_positions"] > 0 and not rep["zero_or_missing_gradients"]
                     and rep["input_shape"][1] == 5 and rep["input_rows_sum_to_one"] and p_ok)
    return rep


def p_out_checks(head: CalibratorHead, init_path: Path, prob: np.ndarray, device) -> dict:
    """The identity start returns the clamped, renormalised input; changing a neighbouring frame changes nothing else."""
    model = new_model(head, init_path, device); p = np.asarray(prob, np.float32); q = head.probs(model, p, device)
    c = np.clip(p, head.clamp_min, None); ref = c / c.sum(1, keepdims=True)
    alt = p.copy(); t = p.shape[0] // 2; alt[t] = np.roll(alt[t], 1); q2 = head.probs(model, alt, device)
    keep = np.ones(p.shape[0], bool); keep[t] = False
    return {"identity_max_abs": float(np.abs(q - ref).max()), "neighbour_change_max_abs": float(np.abs(q2[keep] - q[keep]).max())}


# ----------------------------------------------------------------------------------------------- training
def train_one(cfg: dict, arm: str, seed: int, paths, device) -> dict:
    root = paths.repo_root; s = Setup(cfg, arm, seed, paths, device); tr = s.tr; key = f"{arm}_s{seed}"
    run_dir = root / "runs" / NS / "train" / key; ckpt = root / "checkpoints" / NS / key; done = finished_run(run_dir / "run_manifest.json", s.ident)
    if done is True:
        print(f"{key}: finished with this identity, nothing to do", flush=True); return json.loads((run_dir / "run_manifest.json").read_text())
    if done is False:
        raise SystemExit(f"{key}: a finished run of another identity exists in {run_dir}; move it before training again")
    if (ckpt / "resume.pt").is_file() and torch.load(ckpt / "resume.pt", map_location="cpu", weights_only=False)["identity"] != s.ident:
        raise SystemExit(f"{key}: {ckpt / 'resume.pt'} belongs to another identity; it is never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    model = new_model(s.head, s.init_path, device); opt, sched = make_optimiser(model, tr); start, records = 0, []
    seed_everything(seed)                                   # the dropout masks of T-in and T-out of a seed start from one state
    if (ckpt / "resume.pt").is_file():
        start, records = load_resume(ckpt / "resume.pt", model, opt, sched, s.ident); print(f"{key}: resumed after epoch {start - 1}", flush=True)
    clip = float(tr["grad_clip"]); log = run_dir / "train_log.jsonl"
    if start == 0:
        log.write_text("")
    print(json.dumps({"run": key, "identity": s.ident[:16], "inputs": s.input_keys, "class_weight": s.plan["class_weight"], "pos_weight": s.plan["pos_weight"]}), flush=True)
    for epoch in range(start, int(tr["epochs"])):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time(); sums = defaultdict(float); bs = s.batches(epoch)
        for x in bs:
            for k, v in update(s.head, model, x, opt, sched, clip, device).items():
                sums[k] += v
        rec = {"epoch": epoch, "updates": (epoch + 1) * len(bs), "lr": float(sched.get_last_lr()[0]), "loss": {k: v / len(bs) for k, v in sums.items()},
               "seconds": time.time() - t0, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None}
        records.append(rec)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{key} epoch {epoch} loss {rec['loss']['total']:.4f} ({rec['seconds']:.0f}s)", flush=True)
        save_resume(ckpt / "resume.pt", model, opt, sched, epoch, s.ident, records)
    if len(records) != int(tr["epochs"]):
        raise SystemExit(f"{key}: {len(records)} epochs recorded")
    torch.save(model.state_dict(), ckpt / "final.pt"); (ckpt / "resume.pt").unlink(missing_ok=True)
    man = {"identity": s.ident, "arm": arm, "seed": seed, "epochs": len(records), "updates": records[-1]["updates"], "checkpoint_rule": "final epoch",
           "final_checkpoint": str(ckpt / "final.pt"), "final_sha256": sha256_file(ckpt / "final.pt"), "plan": str(s.plan_path), "plan_sha256": s.plan_sha,
           "init": str(s.init_path), "init_sha256": s.init_sha, "class_weight": s.plan["class_weight"], "pos_weight": s.plan["pos_weight"], "input_keys": s.input_keys,
           "input_content": s.input_content,
           "asrf_commit": getattr(s.head, "commit", None), "records": records}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"run": key, "final_sha256": man["final_sha256"][:16], "minutes": round(sum(r["seconds"] for r in records) / 60, 1)}), flush=True)
    return man


def queue_order(fo: dict) -> list[tuple[str, int]]:
    return [(arm, int(seed)) for seed in fo["training"]["seeds"] for arm in pf.ARMS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=pf.ARMS); ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--queue", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fo = cfg["fold_output_correction"]; paths = load_paths(); root = paths.repo_root
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.queue:
        for arm, seed in queue_order(fo):
            train_one(cfg, arm, seed, paths, device)
        return
    if a.arm is None or a.seed not in [int(s) for s in fo["training"]["seeds"]]:
        raise SystemExit("--arm and one of the round's seeds are required (or --queue)")
    if a.smoke:
        s = Setup(cfg, a.arm, a.seed, paths, device); rep = smoke(s, device)
        out = root / "runs" / NS / "smoke" / f"{a.arm}_s{a.seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**rep, "identity": s.ident, "plan_sha256": s.plan_sha, "init_sha256": s.init_sha, "inputs": s.input_keys}, indent=1)); print(json.dumps(rep, indent=1))
        if not rep["ok"]:
            raise SystemExit("smoke checks failed")
        return
    train_one(cfg, a.arm, a.seed, paths, device)


if __name__ == "__main__":
    main()
