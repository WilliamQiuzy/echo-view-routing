"""Training stages 1 and 2: the window model on EV9V training recordings, update by update from a sampling plan.

    echoview-train --config configs/stage1.json --seed 0 --images /data/EV9V/Images --encoder echo_prime_encoder.pt --out runs/stage1_s0
    echoview-train --config configs/stage2.json --seed 0 --images /data/EV9V/Images --init runs/stage1_s0/final.pt --out runs/stage2_s0

Update u reads the eight windows of row u of the seed's plan: 16 frames each, decoded from the JPEG files, with the
plan's gamma on some fragments and no other change. Stage 1 minimises the cross-entropy at the marked frame. Stage 2
adds the smoothing penalty between the four neighbouring windows of a group (`within_view`). During the first tenth
of the updates only the marker stem and the five-class head train; then the encoder joins.

Everything is plain float32 with deterministic GPU kernels, and one seed fixes the model's dropout. On the machine and
software of the published runs this loop gives their training log number for number (docs/TRAINING.md says how that
was checked). On other hardware the numbers differ in the last digits.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must be set before CUDA starts

import argparse
import json
import time
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..evaluation.ev9v import DataError, folders_by_key, load_frame
from ..model import N_SLOTS, WindowModel, marker_planes, normalise, read_state
from .train import initial_model, lr_factor

GROUP = 4  # neighbouring windows in one group of the within_view objective
TRUNCATION = 16.0
DEFAULTS = {"plan": "windows", "objective": "ce", "penalty": 0.15, "updates": 3750, "warmup": 375, "lr_enc": 1e-5, "lr_new": 1e-4,
            "weight_decay": 1e-4, "grad_clip": 1.0, "log_every": 375, "workers": 6}
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


def deterministic() -> None:
    """Plain float32 and deterministic kernels, as in the published runs."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------------------------------- data
def packaged_plan(kind: str, seed: int) -> Path:
    """The plan of one seed that ships with the package: `windows` (stage 1) or `groups` (stage 2)."""
    return Path(str(resources.files("echoview").joinpath("data", "plans", f"{kind}_s{int(seed)}.npz")))


def read_plan(path: str | Path) -> dict[str, np.ndarray]:
    """recordings (keys), and per update and window: vid, frame and gamma of the 16 slots, the marker slot, the label."""
    with np.load(path, allow_pickle=False) as stored:
        plan = {name: stored[name] for name in ("recordings", "vid", "frame", "gamma", "marker", "target")}
    if plan["vid"].ndim != 3 or plan["vid"].shape[2] != N_SLOTS or plan["frame"].shape != plan["vid"].shape:
        raise ValueError(f"{path}: a plan holds (updates, windows, 16) slots")
    return plan


def frame_files(folder: Path) -> list[Path]:
    """The frame files of a recording in temporal order (numeric file names)."""
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in FRAME_SUFFIXES and p.stem.isdigit()), key=lambda p: int(p.stem))
    if not files:
        raise DataError(f"no frames under {folder}")
    return files


class PlanFrames(Dataset):
    """Item i: the windows of one update as uint8 frames (W, 16, 224, 224, 3), their marker slots and their labels."""

    def __init__(self, plan: dict[str, np.ndarray], images: str | Path, updates: list[int]) -> None:
        self.plan, self.images, self.updates = plan, Path(images), list(updates)
        keys = [str(key) for key in plan["recordings"]]
        used = sorted({int(v) for v in np.unique(plan["vid"])})
        names = folders_by_key(self.images, {keys[v] for v in used})
        self.folders = {v: self.images / names[keys[v]] for v in used}
        self._files: dict[int, list[Path]] = {}  # filled per worker process

    def __len__(self) -> int:
        return len(self.updates)

    def files(self, vid: int) -> list[Path]:
        if vid not in self._files:
            self._files[vid] = frame_files(self.folders[vid])
        return self._files[vid]

    def __getitem__(self, i: int) -> tuple:
        u = self.updates[i]
        vid, frame, gamma = self.plan["vid"][u], self.plan["frame"][u], self.plan["gamma"][u]
        windows = vid.shape[0]
        frames = np.empty((windows, N_SLOTS, 224, 224, 3), np.uint8)
        seen: dict[tuple, np.ndarray] = {}
        for w in range(windows):
            for slot in range(N_SLOTS):
                ref = (int(vid[w, slot]), int(frame[w, slot]), round(float(gamma[w, slot]), 6))
                if ref not in seen:
                    files = self.files(ref[0])
                    seen[ref] = load_frame(files[min(ref[1], len(files) - 1)], ref[2])
                frames[w, slot] = seen[ref]
        return (u, torch.from_numpy(frames), torch.from_numpy(self.plan["marker"][u].astype(np.int64)),
                torch.from_numpy(self.plan["target"][u].astype(np.int64)))


def loader(plan: dict[str, np.ndarray], images: str | Path, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(PlanFrames(plan, images, updates), batch_size=None, num_workers=workers,
                      prefetch_factor=2 if workers else None, pin_memory=True, persistent_workers=False)


# ----------------------------------------------------------------------------------------------- objective
def within_view_loss(logits: torch.Tensor, target: torch.Tensor, penalty: float, group: int = GROUP) -> tuple[torch.Tensor, dict]:
    """Cross-entropy plus `penalty` times the smoothing term, accumulated in float64 from the float32 logits.

    The smoothing term is, for every group of neighbouring windows, the squared difference between each window's
    log-probabilities and the previous window's (held constant), cut at 16 per term, averaged over the three edges and
    the classes, then over the groups. The two terms take the log-softmax separately, as the published runs did.
    """
    classes = logits.shape[-1]
    z = logits.view(-1, group, classes)
    log_p = F.log_softmax(z, dim=-1).to(torch.float64)
    errors = -log_p.gather(-1, target.view(-1, group, 1)).squeeze(-1)
    weights = torch.ones_like(errors)
    ce = (weights * errors).sum() / weights.sum()
    ell = F.log_softmax(z, dim=-1).to(torch.float64)
    d = torch.clamp((ell[:, 1:] - ell[:, :-1].detach()) ** 2, min=0.0, max=TRUNCATION)
    smooth = (d.sum(dim=(1, 2)) / float(d.shape[1] * d.shape[2])).mean()
    loss = ce + float(penalty) * smooth
    return loss, {"ce": float(ce.detach()), "smoothing": float(smooth.detach())}


# ------------------------------------------------------------------------------------------------ the loop
def schedule(update: int, warmup: int, total: int, lr_enc: float, lr_new: float) -> tuple[float, float]:
    """(encoder rate, rate of the new parameters) of one update; the encoder rests during warm-up."""
    factor = lr_factor(int(update), int(warmup), int(total))
    return (0.0 if update < warmup else float(lr_enc) * factor), float(lr_new) * factor


def train_step(model: WindowModel, optimiser: torch.optim.Optimizer, batch: tuple, args: argparse.Namespace, device: torch.device) -> dict:
    _, frames, slots, target = batch
    model.train()
    logits = model(normalise(frames.to(device, non_blocking=True)), marker_planes(slots.to(device)))
    if args.objective == "within_view":
        loss, _parts = within_view_loss(logits, target.to(device), args.penalty)
    else:
        loss = F.cross_entropy(logits, target.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for g in optimiser.param_groups for p in g["params"] if p.grad is not None]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(args.grad_clip)))
    if not np.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    optimiser.step()
    return {"loss": float(loss.detach()), "grad_norm": norm, "correct": int((logits.detach().argmax(1).cpu() == target).sum()),
            "windows": int(target.numel())}


def start_model(args: argparse.Namespace) -> WindowModel:
    """Stage 1 starts from the pretrained encoder (--encoder), stage 2 from a whole window model (--init)."""
    if args.init:
        model = initial_model(None, 0)
        model.load_state_dict(read_state(args.init), strict=True)
        return model
    if not args.encoder:
        raise SystemExit("give --init (a checkpoint of the window model) or --encoder (the pretrained encoder's state)")
    return initial_model(args.encoder, args.seed)


def epoch_record(steps: list[dict], done: int, every: int, began: float, rates: tuple[float, float], model: WindowModel) -> dict:
    return {"epoch": done // every - 1, "updates": done, "loss": float(np.mean([s["loss"] for s in steps])),
            "train_accuracy": sum(s["correct"] for s in steps) / sum(s["windows"] for s in steps),
            "grad_norm_mean": float(np.mean([s["grad_norm"] for s in steps])), "grad_norm_max": float(max(s["grad_norm"] for s in steps)),
            "lr_encoder_last": rates[0], "lr_new_last": rates[1], "seconds": time.time() - began,
            "marker_abs_max": float(model.marker_stem.weight.detach().abs().max())}


def train(args: argparse.Namespace) -> Path:
    """Runs the updates and returns the folder with the results."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deterministic()
    device = torch.device(args.device)
    plan = read_plan(args.plan_file or packaged_plan(args.plan, args.seed))
    if args.objective == "within_view" and plan["vid"].shape[1] % GROUP:
        raise SystemExit("the within_view objective needs whole groups of four windows per update")
    total = min(int(args.updates), plan["vid"].shape[0], int(args.stop_after) or int(args.updates))
    model = start_model(args).to(device)
    optimiser = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": 0.0}, {"params": model.new_parameters(), "lr": 0.0}],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=float(args.weight_decay))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    record = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    (out / "config.json").write_text(json.dumps({**record, "updates_run": total, "torch": torch.__version__}, indent=1) + "\n")
    began, steps, rates = time.time(), [], (0.0, 0.0)
    with (out / "train_log.jsonl").open("a") as log:
        for batch in loader(plan, args.images, list(range(total)), args.workers):
            update = int(batch[0])
            rates = schedule(update, args.warmup, args.updates, args.lr_enc, args.lr_new)
            model.set_encoder_trainable(update >= args.warmup)
            optimiser.param_groups[0]["lr"], optimiser.param_groups[1]["lr"] = rates
            steps.append(train_step(model, optimiser, batch, args, device))
            done = update + 1
            if done % args.log_every == 0 or done == total:
                entry = epoch_record(steps, done, args.log_every, began, rates, model)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in entry.items()}), flush=True)
                steps = []
    torch.save(model.state_dict(), out / "final.pt")
    (out / "done.json").write_text(json.dumps({"updates": total, "seconds": time.time() - began}) + "\n")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options from a recipe file (--config) with the command line on top."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default="")
    known, _ = first.parse_known_args(argv)
    recipe = {**DEFAULTS, **(json.loads(Path(known.config).read_text()) if known.config else {})}
    unknown = sorted(set(recipe) - set(DEFAULTS) - {"description", "loop"})
    if unknown:
        raise SystemExit(f"{known.config}: unknown options {unknown}")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="", help="a recipe file (JSON); the command line wins")
    parser.add_argument("--images", required=True, help="the Images folder of EV9V")
    parser.add_argument("--out", required=True, help="folder for the results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", default="", help="stage 2: a checkpoint of the window model to continue from")
    parser.add_argument("--encoder", default="", help="stage 1: the pretrained encoder's state")
    parser.add_argument("--plan", choices=("windows", "groups"), help="which packaged plan of the seed to follow")
    parser.add_argument("--plan-file", default="", help="a plan file of your own instead of the packaged one")
    parser.add_argument("--objective", choices=("ce", "within_view"))
    parser.add_argument("--penalty", type=float, help="weight of the smoothing term of within_view")
    parser.add_argument("--updates", type=int, help="the length of the schedule; the plan's rows are followed in order")
    parser.add_argument("--warmup", type=int, help="updates during which the encoder rests and the rates warm up")
    parser.add_argument("--lr-enc", type=float)
    parser.add_argument("--lr-new", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--log-every", type=int, help="updates per record of train_log.jsonl")
    parser.add_argument("--workers", type=int, help="processes that decode frames")
    parser.add_argument("--stop-after", type=int, default=0, help="stop after this many updates while the schedule still spans --updates (checks)")
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(**{key: value for key, value in recipe.items() if key not in ("description", "loop")})
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU (--device cpu runs, but only for a look at a few updates)")
    try:
        out = train(args)
    except DataError as error:
        raise SystemExit(str(error)) from None
    print(f"wrote {out / 'final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
