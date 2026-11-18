"""Training the five-view window model on EV9V training recordings.

    echoview-train --config configs/stage3.json --seed 0 --cache /dev/shm/ev9v_cache --init stage2_s0.pt --out runs/C2_s0

One update reads `batch` windows of 16 grey frames from the frame cache, changes them at random on the GPU
(augment.py) and takes one AdamW step. The objective is cross-entropy at the marked frame; `within_view` adds the
smoothing penalty between the four neighbouring windows of a group. Learning rates warm up linearly and then follow a
cosine to zero. The run writes config.json, train_log.jsonl, final.pt (the last weights, the ones every published
number uses) and ema.pt (their moving average). A CUDA GPU is needed; the reference recipe uses 31 GB at batch 32.

Only EV9V training recordings are read. A run is reproducible in what it samples (the windows and the random input
changes follow from the seed) but not bit for bit: GPU kernels and compiled code differ between machines.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import asdict
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..model import EMBED_DIM, WindowModel, marker_planes, read_state
from . import augment, plans
from .cache import Cache
from .fast import Average, Direct

GROUPED = ("within_view",)  # the objectives that train on groups of four neighbouring windows
SAMPLING_STREAM = 20261112  # first word of the seeds of the window sampler and of the augmentation
DEFAULTS = {
    "updates": 3000, "batch": 64, "lr_enc": 3e-5, "lr_new": 3e-4, "warmup": 300, "freeze": 1, "weight_decay": 1e-4,
    "strength": 1.0, "geometry": 1.0, "levels": 1.0, "degrade": 1.0, "shared": 0.5, "native_share": 0.5,
    "label_smoothing": 0.0, "ema": 0.998, "amp": 1, "objective": "ce", "penalty": 0.15, "frames": "gpu", "compile": 0,
    "log_every": 50,
}


def lr_factor(update: int, warmup: int, total: int) -> float:
    """Linear warm-up, then a cosine to zero."""
    if update < warmup:
        return (update + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / max(1, total - warmup)))


def neighbour_penalty(logits: torch.Tensor, group: int = plans.GROUP, truncation: float = 16.0) -> torch.Tensor:
    """The smoothing penalty of the `within_view` objective (the truncated smoothing loss of MS-TCN, Farha and Gall
    2019): for each group of neighbouring windows, the squared difference between each window's log-probabilities and
    the previous window's, which is held constant. Each term is cut at `truncation`. The sum over the three edges and
    the classes is divided by their number and averaged over the groups."""
    classes = logits.shape[-1]
    ell = F.log_softmax(logits.float(), -1).view(-1, group, classes)
    d = (ell[:, 1:] - ell[:, :-1].detach()).pow(2).clamp(max=truncation)
    return d.sum((1, 2)).div((group - 1) * classes).mean()


def initial_model(encoder: str | Path | None, seed: int) -> WindowModel:
    """A window model at the start of training from a pretrained encoder: the marker stem is zero, so the model first
    behaves as if no frame were marked, and the five-class head is drawn from N(0, 0.02^2) with the seed's generator.
    `encoder` is the state of the video encoder with its 512-d projection; None leaves it randomly initialised."""
    model = WindowModel()
    if encoder is not None:
        model.encoder.load_state_dict(read_state(encoder), strict=True)
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        model.marker_stem.weight.zero_()
        model.classifier.weight.copy_(torch.randn(len(model.classifier.bias), EMBED_DIM, generator=generator) * 0.02)
        model.classifier.bias.zero_()
    return model


def build_model(init: str | Path, encoder: str | Path | None, seed: int) -> WindowModel:
    """`init` is a checkpoint of the whole window model, or the word `encoder` to start from the pretrained encoder."""
    if str(init) == "encoder":
        if encoder is None:
            raise SystemExit("--init encoder needs --encoder, the file with the pretrained encoder's state")
        return initial_model(encoder, seed)
    model = initial_model(None, seed)
    model.load_state_dict(read_state(init), strict=True)
    return model


class Feeder(threading.Thread):
    """Draws the windows of every update in order, ahead of the training loop. When the frames stay in host memory it
    also gathers them, so that several training processes can share one copy of the frame array."""

    def __init__(self, pool, rng: np.random.Generator, batch: int, native_share: float, frames, depth: int = 4) -> None:
        super().__init__(daemon=True)
        self.pool, self.rng, self.batch, self.native_share, self.frames = pool, rng, batch, native_share, frames
        self.ready: queue.Queue = queue.Queue(maxsize=depth)

    def step(self) -> tuple:
        """The windows of the next update and, when the frames stay in host memory, their grey frames."""
        rows, marker, target, fragment = self.pool.batch(self.rng, self.batch, self.native_share)
        source = None
        if self.frames is not None:
            flat = self.frames[rows.reshape(-1)]
            source = torch.from_numpy(flat).view(len(rows), plans.SLOTS, *flat.shape[1:])
        return rows, marker, target, fragment, source

    def run(self) -> None:
        try:
            while True:
                self.ready.put(self.step())
        except Exception as error:
            self.ready.put(error)


def packaged_plans(objective: str) -> list[Path]:
    """The plan files that ship with the package: the groups for `within_view`, the single windows for `ce`. Every
    seed draws from the plans of all three."""
    kind = "groups" if objective in GROUPED else "windows"
    folder = resources.files("echoview").joinpath("data", "plans")
    return [Path(str(folder.joinpath(f"{kind}_s{seed}.npz"))) for seed in (0, 1, 2)]


def set_rates(optimiser: torch.optim.Optimizer, args: argparse.Namespace, update: int, frozen: bool) -> None:
    """The learning rates of this update: group 0 is the encoder, group 1 the new parameters."""
    factor = lr_factor(update, args.warmup, args.updates)
    optimiser.param_groups[0]["lr"] = 0.0 if frozen else args.lr_enc * factor
    optimiser.param_groups[1]["lr"] = args.lr_new * factor


def run_config(args: argparse.Namespace, strength: augment.Strength, pool) -> dict:
    """The record of the run that goes into config.json."""
    record = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    record["plans"] = [str(path) for path in args.plans]
    return {**record, "strength_ranges": asdict(strength), "pool_rows": len(pool.rows), "train_rows": int(pool.train_end),
            "torch": torch.__version__}


def train(args: argparse.Namespace) -> Path:
    """Runs the updates and returns the folder with the results."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    cache = Cache(args.cache)
    grouped = args.objective in GROUPED
    args.plans = [Path(path) for path in args.plans] if args.plans else packaged_plans(args.objective)
    pool = (plans.GroupPool if grouped else plans.Pool)(cache, args.plans)
    strength = augment.Strength(scale=args.strength, geometry=args.geometry, levels=args.levels, degrade=args.degrade,
                                shared=1.0 if grouped else args.shared)
    (out / "config.json").write_text(json.dumps(run_config(args, strength, pool), indent=1) + "\n")
    frames = torch.from_numpy(np.ascontiguousarray(cache.frames[:pool.train_end])).to(device) if args.frames == "gpu" else None
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(args.init, args.encoder, args.seed).to(device).train()
    optimiser = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": 0.0}, {"params": model.new_parameters(), "lr": 0.0}],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay, fused=True)
    forward = torch.compile(Direct(model)) if args.compile else Direct(model)
    rng = np.random.default_rng([SAMPLING_STREAM, args.seed])
    generator = torch.Generator(device).manual_seed(SAMPLING_STREAM + args.seed)
    average = Average(model)
    feeder = Feeder(pool, rng, args.batch, args.native_share, cache.frames if args.frames == "host" else None)
    feeder.start()
    frozen = None
    began = last = time.time()
    seen = correct = 0
    running = 0.0
    with (out / "train_log.jsonl").open("a") as log:
        for update in range(args.updates):
            want_frozen = bool(args.freeze) and update < args.warmup
            if want_frozen != frozen:
                model.set_encoder_trainable(not want_frozen)
                frozen = want_frozen
            set_rates(optimiser, args, update, frozen)
            item = feeder.ready.get()
            if isinstance(item, Exception):
                raise item
            rows, marker, target, fragment, source = item
            planes = marker_planes(torch.from_numpy(marker).to(device))
            with torch.no_grad():
                source = frames[torch.from_numpy(rows).to(device)] if source is None else source.to(device, non_blocking=True)
                rgb = augment.augment(source, torch.from_numpy(fragment).to(device), strength, generator, plans.GROUP if grouped else 1)
            labels = torch.from_numpy(target).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                logits = forward(rgb, planes)
            loss = F.cross_entropy(logits.float(), labels, label_smoothing=args.label_smoothing)
            if grouped:
                loss = loss + args.penalty * neighbour_penalty(logits)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimiser.param_groups for p in g["params"] if p.grad is not None], 1.0)
            optimiser.step()
            average.update(args.ema if not frozen else 0.0)
            seen += args.batch
            correct += int((logits.argmax(-1) == labels).sum())
            running += float(loss)
            if (update + 1) % args.log_every == 0 or update + 1 == args.updates:
                now = time.time()
                record = {"update": update + 1, "loss": running / args.log_every, "accuracy": correct / (args.log_every * args.batch),
                          "windows_per_s": args.log_every * args.batch / (now - last), "lr_enc": optimiser.param_groups[0]["lr"],
                          "lr_new": optimiser.param_groups[1]["lr"], "gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "elapsed_s": now - began}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in record.items()}), flush=True)
                last, correct, running = now, 0, 0.0
    torch.save(model.state_dict(), out / "final.pt")
    torch.save(average.state_dict(model), out / "ema.pt")
    (out / "done.json").write_text(json.dumps({"updates": args.updates, "windows": seen, "seconds": time.time() - began}) + "\n")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options from a recipe file (--config) with the command line on top."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default="")
    known, _ = first.parse_known_args(argv)
    recipe = {**DEFAULTS, **(json.loads(Path(known.config).read_text()) if known.config else {})}
    unknown = sorted(set(recipe) - set(DEFAULTS) - {"description"})
    if unknown:
        raise SystemExit(f"{known.config}: unknown options {unknown}")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="", help="a recipe file (JSON) with any of the options below; the command line wins")
    parser.add_argument("--cache", required=True, help="the frame cache folder (python -m echoview.training.cache)")
    parser.add_argument("--plans", nargs="+", default=None, help="plan files of your own (default: the plans that ship with the package)")
    parser.add_argument("--out", required=True, help="folder for the results")
    parser.add_argument("--init", required=True, help="a checkpoint of the window model, or the word `encoder`")
    parser.add_argument("--encoder", default=None, help="with --init encoder: the pretrained encoder's state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch", type=int, help="windows per update (whole groups of four for within_view)")
    parser.add_argument("--lr-enc", type=float, help="learning rate of the encoder")
    parser.add_argument("--lr-new", type=float, help="learning rate of the marker stem and the five-class head")
    parser.add_argument("--warmup", type=int, help="updates of linear warm-up")
    parser.add_argument("--freeze", type=int, help="1: the encoder stays frozen during warm-up; 0: it trains from the start")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--strength", type=float, help="multiplier of every random input change (0: none)")
    parser.add_argument("--geometry", type=float, help="multiplier of the geometric changes")
    parser.add_argument("--levels", type=float, help="multiplier of the grey-level changes")
    parser.add_argument("--degrade", type=float, help="multiplier of blur, noise and dark borders")
    parser.add_argument("--shared", type=float, help="probability that the fragments of a window share one set of changes")
    parser.add_argument("--native-share", type=float, help="ce only: share of the windows drawn inside one recording")
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--ema", type=float, help="decay of the moving average of the weights")
    parser.add_argument("--amp", type=int, help="1: bfloat16 autocast")
    parser.add_argument("--objective", choices=("ce", "within_view"))
    parser.add_argument("--penalty", type=float, help="weight of the smoothing penalty of within_view")
    parser.add_argument("--frames", choices=("gpu", "host"), help="where the frame array lives; host lets several runs share it")
    parser.add_argument("--compile", type=int, help="1: torch.compile the forward pass")
    parser.add_argument("--log-every", type=int)
    parser.set_defaults(**{key: value for key, value in recipe.items() if key != "description"})
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU")
    out = train(args)
    print(f"wrote {out / 'final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
