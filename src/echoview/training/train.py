"""Training the five-view window model on EV9V training recordings.

    echoview-train --config configs/stage3.json --seed 0 --cache /dev/shm/ev9v_cache --init stage2_s0.pt --out runs/C2_s0

One update reads `batch` windows of 16 grey frames from the frame cache, changes them at random on the GPU
(augment.py) and takes one AdamW step. The objective is cross-entropy at the marked frame; `within_view` adds the
smoothing penalty between the four neighbouring windows of a group. Learning rates warm up linearly and then follow a
cosine to zero. The run writes config.json, train_log.jsonl, final.pt (the last weights, the ones every published
number uses) and ema.pt (their moving average). A CUDA GPU is needed; the reference recipe uses 31 GB at batch 32.

Only EV9V training recordings are read. A run is reproducible in what it samples: the windows follow from the seed,
and so do the random input changes on the same GPU model. It is not reproducible bit for bit: GPU kernels and compiled
code differ between machines and between runs.

On a machine that can be switched off, `--checkpoint-every N` writes checkpoint.pt every N updates and `--resume`
continues from it: weights, optimiser, moving average and every random stream go on where they stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import weights
from ..model import EMBED_DIM, WindowModel, marker_planes, read_state
from . import augment, plans, recipe
from .cache import Cache
from .fast import Average, Direct

GROUPED = ("within_view",)  # the objectives that train on groups of four neighbouring windows
SAMPLING_STREAM = 20261112  # first word of the seeds of the window sampler and of the augmentation
DEFAULTS = {
    "updates": 3000, "batch": 64, "lr_enc": 3e-5, "lr_new": 3e-4, "warmup": 300, "freeze": 1, "weight_decay": 1e-4,
    "strength": 1.0, "geometry": 1.0, "levels": 1.0, "degrade": 1.0, "shared": 0.5, "native_share": 0.5, "mirror": 0.0,
    "label_smoothing": 0.0, "ema": 0.998, "amp": 1, "objective": "ce", "penalty": 0.15, "frames": "gpu", "compile": 0,
    "log_every": 50, "checkpoint_every": 0,
}
CHECKPOINT = "checkpoint.pt"
NOT_THE_RECIPE = ("checkpoint_every",)  # may change between the parts of one run


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
    """`init` is a checkpoint of the whole window model, the word `stage2` (the published stage-2 model of the seed,
    downloaded and checked) or the word `encoder` to start from the pretrained encoder."""
    if str(init) == "encoder":
        if not encoder:
            raise SystemExit("--init encoder needs --encoder, the file with the pretrained encoder's state")
        return initial_model(encoder, seed)
    try:
        start = weights.start_weights(seed) if str(init) == "stage2" else init
    except weights.WeightsError as error:
        raise SystemExit(str(error)) from None
    model = initial_model(None, seed)
    model.load_state_dict(read_state(start), strict=True)
    return model


def check_start(args: argparse.Namespace, continuing: bool = False) -> None:
    """Mistakes that would otherwise show only after the frame cache has been loaded. A run that continues from a
    checkpoint does not read its start weights again, so they need not be there."""
    if args.objective in GROUPED and args.batch % plans.GROUP:
        raise SystemExit(f"--batch {args.batch}: the within_view objective needs whole groups of {plans.GROUP} windows")
    if not 0.0 <= args.mirror <= 1.0:
        raise SystemExit(f"--mirror {args.mirror}: a probability, from 0 to 1")
    if continuing:
        pass
    elif str(args.init) == "encoder":
        if not args.encoder or not Path(args.encoder).is_file():
            raise SystemExit("--init encoder needs --encoder, the file with the pretrained encoder's state")
    elif str(args.init) != "stage2" and not Path(args.init).is_file():
        raise SystemExit(f"{args.init}: no such start file (a checkpoint of the window model, or one of the words stage2 and encoder)")
    if args.checkpoint_every and args.checkpoint_every % args.log_every:
        raise SystemExit(f"--checkpoint-every {args.checkpoint_every} must be a multiple of --log-every {args.log_every}: "
                         "a checkpoint is written together with a log record")


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


# ------------------------------------------------------------------------------- stopping and going on
PARTS = ("updates_done", "identity", "model", "optimiser", "average", "generator", "torch_rng", "cuda_rng", "seen", "elapsed_s")


def write_whole(path: Path, write) -> None:
    """Writes a file beside its place, forces it to disk and swaps it in: switching the machine off leaves the old
    file or the new one, never half of one. `write` receives the open binary file."""
    part = path.with_name(path.name + ".tmp")
    with part.open("wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def file_name(path: object) -> str:
    """A file by its name alone: the same file reached by another path (a new machine, another mount) is the same."""
    return Path(str(path)).name


def run_identity(args: argparse.Namespace) -> dict:
    """The options that make two parts one run: the recipe, the seed, the start weights and the plans.

    Stored runs are held to this identity (a checkpoint, and docs/ablations/arms.json, which cannot be made again without
    training). An option that is added later must therefore leave the identity of a run without it as it was: it is
    dropped where it is unset, as the mirror is."""
    identity = {key: getattr(args, key) for key in sorted(DEFAULTS) if key not in NOT_THE_RECIPE}
    if not identity["mirror"]:  # a run without a mirror is the run it was before the option existed
        del identity["mirror"]
    return {**identity, "seed": int(args.seed), "init": file_name(args.init),
            "plans": [file_name(path) for path in args.plans] if args.plans else None}


def same_run(stored: dict, now: dict, path: Path) -> None:
    differing = sorted(key for key in set(stored) | set(now) if stored.get(key) != now.get(key))
    if differing:
        changes = "; ".join(f"{key}: {stored.get(key)} -> {now.get(key)}" for key in differing)
        raise SystemExit(f"{path} belongs to a run with other options ({changes}). Give the options of that run to continue it, "
                         "or use another folder for the new run.")


def resume_note(state: dict | None, args: argparse.Namespace, out: Path) -> str:
    """One line that says where a run started with --resume begins."""
    if state is not None:
        return f"{out}: continuing after update {int(state['updates_done'])} of {args.updates}"
    note = f"{out}: no checkpoint, starting at the first update"
    if not args.checkpoint_every:
        note += " (without --checkpoint-every this run writes no checkpoint to continue from)"
    return note


def skip_draws(pool, rng: np.random.Generator, updates: int, batch: int, native_share: float) -> None:
    """Lets the sampler draw again what `updates` finished updates drew, so that the next draw is the next update's."""
    for _ in range(updates):
        pool.batch(rng, batch, native_share)


def keep_log_until(log: Path, done: int) -> None:
    """Drops the records a cut run wrote after its last checkpoint, and a half-written last line."""
    kept = []
    if log.is_file():
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("update", 0)) <= done:
                kept.append(line)
    write_whole(log, lambda handle: handle.write("".join(line + "\n" for line in kept).encode()))


def save_checkpoint(out: Path, done: int, identity: dict, model: WindowModel, optimiser: torch.optim.Optimizer, average: Average,
                    generator: torch.Generator, seen: int, elapsed: float, device: torch.device, sessions: int = 1) -> None:
    """Everything the loop needs to go on after update `done`: the weights, the optimiser, the moving average and
    the random streams (the input changes' own, and the ones dropout draws from on the CPU and on the training GPU).
    `sessions` counts the sessions that made the updates up to here."""
    state = {"updates_done": int(done), "identity": identity, "model": model.state_dict(), "optimiser": optimiser.state_dict(),
             "average": average.state(), "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None, "seen": int(seen), "elapsed_s": float(elapsed),
             "sessions": int(sessions)}
    write_whole(out / CHECKPOINT, lambda handle: torch.save(state, handle))


def load_checkpoint(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SystemExit(f"{path} cannot be read: {type(error).__name__}: {str(error)[:300]}. If the file is damaged, move it away: "
                         "the run then starts again at its first update.") from error
    missing = [part for part in PARTS if not isinstance(state, dict) or part not in state]
    if missing:
        raise SystemExit(f"{path} is not a checkpoint of this program (it lacks {', '.join(missing)})")
    return state


def leave_finished(out: Path, identity: dict) -> None:
    """A folder with done.json holds a finished run. It is left alone when it is this run and refused when it is another."""
    try:
        finished = json.loads((out / "done.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{out / 'done.json'} cannot be read ({error}); the folder holds a finished run that cannot be identified") from error
    same_run(finished.get("identity", identity), identity, out / "done.json")
    (out / CHECKPOINT).unlink(missing_ok=True)
    print(f"{out} holds this run, finished: nothing to do", file=sys.stderr)


def train(args: argparse.Namespace) -> Path:
    """Runs the updates and returns the folder with the results."""
    out = Path(args.out)
    identity = run_identity(args)
    if args.resume and (out / "done.json").is_file():
        leave_finished(out, identity)
        return out
    state = load_checkpoint(out / CHECKPOINT) if args.resume and (out / CHECKPOINT).is_file() else None
    if state is not None:
        same_run(state["identity"], identity, out / CHECKPOINT)
    check_start(args, continuing=state is not None)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        print(resume_note(state, args, out), file=sys.stderr)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"  # the CPU runs a few updates for a look or a test, without autocast and fused steps
    cache = Cache(args.cache)
    grouped = args.objective in GROUPED
    args.plans = [Path(path) for path in args.plans] if args.plans else packaged_plans(args.objective)
    pool = (plans.GroupPool if grouped else plans.Pool)(cache, args.plans)
    strength = augment.Strength(scale=args.strength, geometry=args.geometry, levels=args.levels, degrade=args.degrade,
                                shared=1.0 if grouped else args.shared)
    (out / "config.json").write_text(json.dumps(run_config(args, strength, pool), indent=1) + "\n")
    frames = torch.from_numpy(np.ascontiguousarray(cache.frames[:pool.train_end])).to(device) if args.frames == "gpu" else None
    torch.manual_seed(args.seed)
    if on_gpu:
        torch.cuda.manual_seed_all(args.seed)
    model = initial_model(None, args.seed) if state is not None else build_model(args.init, args.encoder, args.seed)  # a checkpoint holds every weight
    model = model.to(device).train()
    optimiser = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": 0.0}, {"params": model.new_parameters(), "lr": 0.0}],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay, fused=on_gpu)
    forward = torch.compile(Direct(model)) if args.compile else Direct(model)
    rng = np.random.default_rng([SAMPLING_STREAM, args.seed])
    generator = torch.Generator(device).manual_seed(SAMPLING_STREAM + args.seed)
    average = Average(model)
    began = last = time.time()
    done = seen = correct = 0
    sessions = 1  # the sessions that the updates of the finished model come from: one, and one more for every checkpoint a run went on from
    if state is not None:  # go on after the last checkpoint: the weights, the averages and every random stream
        done, seen, began = int(state["updates_done"]), int(state["seen"]), began - float(state["elapsed_s"])
        sessions = int(state.get("sessions", 1)) + 1
        model.load_state_dict(state["model"], strict=True)
        optimiser.load_state_dict(state["optimiser"])
        average.load(state["average"])
        skip_draws(pool, rng, done, args.batch, args.native_share)
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        if on_gpu and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    if args.resume:
        keep_log_until(out / "train_log.jsonl", done)
    feeder = Feeder(pool, rng, args.batch, args.native_share, cache.frames if args.frames == "host" else None)
    feeder.start()
    frozen = None
    running = 0.0
    with (out / "train_log.jsonl").open("a") as log:
        for update in range(done, args.updates):
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
                group = plans.GROUP if grouped else 1
                mirror = augment.mirrored(args.seed, update, args.batch // group, args.mirror).repeat_interleave(group) if args.mirror > 0 else None
                rgb = augment.augment(source, torch.from_numpy(fragment).to(device), strength, generator, group, mirror)
            labels = torch.from_numpy(target).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(args.amp) and on_gpu):
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
                          "lr_new": optimiser.param_groups[1]["lr"], "gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0,
                          "elapsed_s": now - began}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in record.items()}), flush=True)
                last, correct, running = now, 0, 0.0
            if args.checkpoint_every and (update + 1) % args.checkpoint_every == 0 and update + 1 < args.updates:
                os.fsync(log.fileno())  # the log's records up to here are on disk before the checkpoint that counts on them
                save_checkpoint(out, update + 1, identity, model, optimiser, average, generator, seen, time.time() - began, device, sessions)
    # the weights first and whole, then done.json, which says the run is finished, then the checkpoint goes
    write_whole(out / "final.pt", lambda handle: torch.save(model.state_dict(), handle))
    write_whole(out / "ema.pt", lambda handle: torch.save(average.state_dict(model), handle))
    finished = {"updates": args.updates, "windows": seen, "seconds": time.time() - began, "sessions": sessions, "identity": identity}
    write_whole(out / "done.json", lambda handle: handle.write((json.dumps(finished) + "\n").encode()))
    (out / CHECKPOINT).unlink(missing_ok=True)
    return out


def count(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("a count cannot be negative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Options from a recipe file (--config) with the command line on top; both are checked in the same way."""
    argv = sys.argv[1:] if argv is None else list(argv)
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default="")
    known, _ = first.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="", help="a recipe file (JSON) with any of the options below; the command line wins")
    parser.add_argument("--cache", required=True, help="the frame cache folder (python -m echoview.training.cache)")
    parser.add_argument("--plans", nargs="+", default=None, help="plan files of your own (default: the plans that ship with the package)")
    parser.add_argument("--out", required=True, help="folder for the results")
    parser.add_argument("--init", required=True,
                        help="a checkpoint of the window model; `stage2` for the published stage-2 model of the seed; or `encoder`")
    parser.add_argument("--device", default="cuda", help="cuda; cpu runs a few updates for a look, without autocast")
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
    parser.add_argument("--mirror", type=float, help="probability that a group of windows is mirrored left to right (0: never, as in every published recipe)")
    parser.add_argument("--native-share", type=float, help="ce only: share of the windows drawn inside one recording")
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--ema", type=float, help="decay of the moving average of the weights")
    parser.add_argument("--amp", type=int, help="1: bfloat16 autocast")
    parser.add_argument("--objective", choices=("ce", "within_view"))
    parser.add_argument("--penalty", type=float, help="weight of the smoothing penalty of within_view")
    parser.add_argument("--frames", choices=("gpu", "host"), help="where the frame array lives; host lets several runs share it")
    parser.add_argument("--compile", type=int, help="1: torch.compile the forward pass")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--checkpoint-every", type=count, help="write checkpoint.pt every this many updates (a multiple of --log-every; 0: never)")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pt in --out when there is one; leave a finished run alone")
    parser.set_defaults(**DEFAULTS)
    return parser.parse_args([*recipe.arguments(known.config, set(DEFAULTS)), *argv])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("training needs a CUDA GPU (--device cpu runs, but only for a look at a few updates)")
    out = train(args)
    print(f"wrote {out / 'final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
