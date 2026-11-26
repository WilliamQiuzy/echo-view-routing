"""The published system on development data that is changed on purpose, and on a view it does not know
(docs/ABLATIONS.md, part 2).

  python -m echoview.evaluation.stress run --images <EV9V Images> --folder <work folder> [--system stage2]
        [--ninth NAME=LIST ...] [--cache DIR]
  python -m echoview.evaluation.stress report --logits <development logits> --folder <work folder> [--stage2 <its work folder>]
        --out forward.json

`run` shows the selection half of the development benchmark to the models once as it is and once under every
condition below, and writes one file per condition into the work folder. A file that exists is not computed again, so
a run that was cut off goes on where it stopped. With `--ninth` it also reads listed recordings of the view that the
system does not know (folder names under Images, one per line), as `echoview classify --fps 30` reads them. The files
hold outputs for single sequences and stay with the data.

`--system stage2` runs the three published stage-2 models, the models from which stage 3 starts, as one ensemble in
place of the published system. It also shows them the unchanged calibration half, because their decoder setting is
chosen there by the project's rule.

`report` runs no model. It decodes every condition, scores it against the unchanged frames with the same draws, and
writes pooled numbers only.

The image conditions change every frame of a sequence in the same way. The four conditions of time change what a
window shows. Nothing of the published system is chosen here, the test benchmark is not read, and nothing changes
the published system.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from .. import frames as fr
from .. import weights as published_weights
from ..model import CLASSES, load_model
from ..pipeline import Ensemble, classification, pick_device, probabilities_from_logits
from ..reference import B3, SEEDS
from ..windows import window_plan
from . import ablations, benchmark, classify_read, comparison, ev9v, followup
from .benchmark import Sequence, of_kind
from .comparison import DESCRIBED, DRAWS, ROUTER, SEED, Row
from .run import member_probabilities

N = len(CLASSES)
SIDE = 224
PUBLISHED, STAGE2 = "published", "stage 2"
CLEAN, STILL, SHUFFLED, REVERSED, HALF_RATE = "unchanged", "still frame", "shuffled frames", "played backwards", "half the frame rate"
TIME = (STILL, SHUFFLED, REVERSED, HALF_RATE)
CALIBRATION_FILE = "calibration_unchanged.npz"  # stage 2 only: the unchanged calibration half, where its decoder setting is chosen
KNOWN_FILE = "known.npz"  # the single recordings of both halves, read as the recordings of the ninth view are read
CLOSE = 1e-3  # the unchanged frames, run again, against the stored logits: probabilities of another run differ in the sixth decimal
NINTH_FPS = 30.0  # EV9V's frames per second, as `echoview classify --fps 30` is given it for a folder of frames
Planes = np.ndarray  # (frames, 224, 224) float64 grey levels


# ------------------------------------------------------------------------------------------- the image conditions
def planes_of(frames: np.ndarray) -> Planes:
    return np.asarray(frames)[..., 0].astype(np.float64)


def frames_of(planes: Planes) -> np.ndarray:
    """Grey levels back to what the models read: rounded, kept in 0 to 255, the same in all three channels."""
    return np.repeat(np.clip(np.rint(planes), 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def _each(planes: Planes, change: Callable) -> Planes:
    """A change that Pillow makes, frame by frame, on rounded grey levels."""
    from PIL import Image

    grey = np.clip(np.rint(planes), 0, 255).astype(np.uint8)
    return np.stack([np.asarray(change(Image.fromarray(frame)), np.float64) for frame in grey])


def _jpeg(image, quality: int):
    from PIL import Image

    held = io.BytesIO()
    image.save(held, format="JPEG", quality=quality)
    held.seek(0)
    return Image.open(held).convert("L")


def _resolution(image, side: int):
    from PIL import Image

    return image.resize((side, side), Image.BILINEAR).resize((SIDE, SIDE), Image.BILINEAR)


def _zoom(image, factor: float):
    """Above 1 the middle of the frame fills it; below 1 the frame is made smaller and set in black."""
    from PIL import Image

    if factor >= 1:
        keep = round(SIDE / factor)
        edge = (SIDE - keep) // 2
        return image.crop((edge, edge, edge + keep, edge + keep)).resize((SIDE, SIDE), Image.BILINEAR)
    small = round(SIDE * factor)
    edge = (SIDE - small) // 2
    ground = Image.new("L", (SIDE, SIDE), 0)
    ground.paste(image.resize((small, small), Image.BILINEAR), (edge, edge))
    return ground


def _rotated(image, degrees: float):
    from PIL import Image

    return image.rotate(degrees, resample=Image.BILINEAR, fillcolor=0)  # about the middle, corners in black


def _blur(planes: Planes, sigma: float) -> Planes:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(planes, sigma=(0.0, sigma, sigma))


IMAGE: dict[str, Callable[[Planes, np.random.Generator], Planes]] = {
    "gamma 0.5": lambda g, rng: 255.0 * (g / 255.0) ** 0.5,
    "gamma 0.7": lambda g, rng: 255.0 * (g / 255.0) ** 0.7,
    "gamma 1.5": lambda g, rng: 255.0 * (g / 255.0) ** 1.5,
    "gamma 2.0": lambda g, rng: 255.0 * (g / 255.0) ** 2.0,
    "darker by 40": lambda g, rng: g - 40.0,
    "brighter by 40": lambda g, rng: g + 40.0,
    "contrast 0.5": lambda g, rng: (g - 127.5) * 0.5 + 127.5,
    "contrast 1.5": lambda g, rng: (g - 127.5) * 1.5 + 127.5,
    "noise 10": lambda g, rng: g + rng.normal(0.0, 10.0, g.shape),
    "noise 25": lambda g, rng: g + rng.normal(0.0, 25.0, g.shape),
    "speckle 0.25": lambda g, rng: g * (1.0 + rng.normal(0.0, 0.25, g.shape)),
    "blur 1 pixel": lambda g, rng: _blur(g, 1.0),
    "blur 2 pixels": lambda g, rng: _blur(g, 2.0),
    "JPEG quality 30": lambda g, rng: _each(g, lambda image: _jpeg(image, 30)),
    "JPEG quality 10": lambda g, rng: _each(g, lambda image: _jpeg(image, 10)),
    "112 pixels": lambda g, rng: _each(g, lambda image: _resolution(image, 112)),
    "64 pixels": lambda g, rng: _each(g, lambda image: _resolution(image, 64)),
    "mirrored": lambda g, rng: g[:, :, ::-1],
    "zoomed in 1.2": lambda g, rng: _each(g, lambda image: _zoom(image, 1.2)),
    "zoomed out 0.8": lambda g, rng: _each(g, lambda image: _zoom(image, 0.8)),
    "rotated 10 degrees": lambda g, rng: _each(g, lambda image: _rotated(image, 10.0)),
    "rotated 20 degrees": lambda g, rng: _each(g, lambda image: _rotated(image, 20.0)),
}
CONDITIONS = (CLEAN, *IMAGE, *TIME)


def generator(name: str, index: int) -> np.random.Generator:
    """One generator for a condition and a sequence, the same in every run and for every system."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode()), index])


# ------------------------------------------------------------------------------------------------ one condition
def shown_still(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Every window shows its centre sample sixteen times."""
    centres, windows, slots = window_plan(n_samples)
    return np.repeat(centres[:, None], windows.shape[1], axis=1), slots


def shown_shuffled(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Every window shows its own sixteen samples in a random order; the marker goes with the centre sample."""
    _, windows, slots = window_plan(n_samples)
    out, marks = np.empty_like(windows), np.empty_like(slots)
    for row, (window, slot) in enumerate(zip(windows, slots)):
        order = rng.permutation(len(window))
        out[row], marks[row] = window[order], int(np.flatnonzero(order == slot)[0])
    return out, marks


def under(system: Ensemble, frames: np.ndarray, name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One sequence under one condition -> the ensemble's probabilities for every sample (samples, 5), in the
    sequence's own order, and the window logits (windows, models, 5)."""
    n = len(frames)
    if name == REVERSED:  # the stream backwards; its outputs are turned round again
        logits = system.window_logits(frames[::-1].copy())
        return probabilities_from_logits(logits, n)[::-1].copy(), logits
    if name == CLEAN:
        logits = system.window_logits(frames)
    elif name in IMAGE:
        logits = system.window_logits(frames_of(IMAGE[name](planes_of(frames), rng)))
    elif name == STILL:
        logits = system.window_logits(frames, shown=shown_still(n))
    elif name == SHUFFLED:
        logits = system.window_logits(frames, shown=shown_shuffled(n, rng))
    elif name == HALF_RATE:  # five frames a second, each shown twice
        logits = system.window_logits(np.ascontiguousarray(frames[(np.arange(n) // 2) * 2]))
    else:
        raise ValueError(f"no such condition: {name}")
    return probabilities_from_logits(logits, n), logits


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save(arrays: Mapping[str, np.ndarray], path: Path) -> None:
    """One .npz file that appears whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, **arrays)
    partial.replace(path)


def run_condition(system: Ensemble, sequences: tuple[Sequence, ...], table: ev9v.FrameTable, name: str) -> dict[str, np.ndarray]:
    """Every sequence under one condition: `p__<id>` holds the per-sample probabilities, and for a single recording
    `c__<id>` holds what `classify` returns (five probabilities and whether the seeds agree)."""
    out = {}
    for index, sequence in enumerate(sequences):
        probabilities, logits = under(system, table.frames(sequence), name, generator(name, index))
        out[f"p__{sequence.id}"] = probabilities
        if sequence.kind == "native":
            decided = classification(logits)
            out[f"c__{sequence.id}"] = np.append(decided.probabilities, float(decided.seeds_agree))
    return out


def read_ninth(system: Ensemble, images: Path, names: list[str]) -> np.ndarray:
    """(recordings, 6): what `classify` returns for every listed recording, read as the command reads a folder."""
    rows = []
    for name in names:
        decided = system.classify(fr.read_recording(images / name, NINTH_FPS).frames)
        rows.append(np.append(decided.probabilities, float(decided.seeds_agree)))
    return np.stack(rows)


def read_known(system: Ensemble, images: Path, halves: Mapping[str, tuple[Sequence, ...]]) -> np.ndarray:
    """(recordings, 6): the single recordings of both development halves through the same reader as the ninth view,
    calibration half first, so that the two kinds of recording are compared on one way of reading a folder."""
    natives = [s for half in benchmark.HALVES for s in of_kind(halves[half], "native")]
    names = ev9v.folders_by_key(images, {s.fragments[0].video for s in natives})
    return read_ninth(system, images, [names[s.fragments[0].video] for s in natives])


def stage2_system(weights_dir: Path | None, device: str, batch: int) -> Ensemble:
    """The three published stage-2 models as one ensemble. The decoder setting it carries is a placeholder: the
    report chooses one for it on the calibration half."""
    where = pick_device(device)
    return Ensemble.from_models([load_model(published_weights.start_weights(seed, weights_dir), where) for seed in SEEDS], B3, device=device, batch=batch)


# ------------------------------------------------------------------------------------------------------ report
def _classified(stored: Mapping[str, np.ndarray], natives: tuple[Sequence, ...]) -> dict:
    """`classify`'s rule on the stored outputs of one condition: wrong recordings, mean probability, seeds that differ."""
    rows = np.stack([stored[f"c__{s.id}"] for s in natives])
    truth = np.array([int(s.labels[0]) for s in natives])
    right = rows[:, :N].argmax(1) == truth
    return {"errors": int((~right).sum()), "mean_probability": float(rows[:, :N].max(1).mean()), "seeds_differ": int((rows[:, N] < 0.5).sum())}


def certainty(rows: np.ndarray) -> dict[str, np.ndarray]:
    """Three numbers that say how sure the five probabilities of a recording are; each is higher when surer."""
    p = np.asarray(rows, float)[:, :N]
    ranked = np.sort(p, axis=1)
    return {"probability": ranked[:, -1], "margin": ranked[:, -1] - ranked[:, -2], "negative_entropy": (p * np.log(np.maximum(p, 1e-12))).sum(1)}


def ninth_part(rows: np.ndarray, known: np.ndarray, draws: int, seed: int) -> dict:
    """Recordings of a view that the system does not know (`rows`), beside recordings of the five views it knows
    (`known`, in the same form): which view they are given, with what probability, and whether a number that says
    how sure the system is tells the two kinds apart."""
    probability, given = rows[:, :N].max(1), rows[:, :N].argmax(1)
    of_known = known[:, :N].max(1)
    is_known = np.concatenate([np.ones(len(known), bool), np.zeros(len(rows), bool)])
    counts = classify_read.drawn(np.arange(len(is_known)), draws, seed)
    sure_known, sure_ninth = certainty(known), certainty(rows)
    return {"recordings": len(rows), "given": {view: int((given == k).sum()) for k, view in enumerate(CLASSES)},
            "probability": {"lowest": float(probability.min()), "quarter": float(np.quantile(probability, 0.25)), "median": float(np.median(probability)),
                            "three_quarters": float(np.quantile(probability, 0.75)), "highest": float(probability.max()),
                            "above_0.9": int((probability > 0.9).sum())},
            "seeds_differ": int((rows[:, N] < 0.5).sum()),
            "known": {"recordings": len(known), "median_probability": float(np.median(of_known)), "above_0.9": int((of_known > 0.9).sum()),
                      "seeds_differ": int((known[:, N] < 0.5).sum())},
            "known_above_ninth": {name: classify_read.separation_interval(np.concatenate([sure_known[name], sure_ninth[name]]), is_known, counts)
                                  for name in sure_known}}


def _held(folder: Path, name: str, sequences: tuple[Sequence, ...]) -> dict[str, np.ndarray]:
    """The file of one condition, which must hold every sequence."""
    path = folder / (CALIBRATION_FILE if name is None else f"{slug(name)}.npz")
    if not path.is_file():
        raise SystemExit(f"{folder} lacks {path.name}")
    with np.load(path) as data:
        natives = of_kind(sequences, "native")
        if any(f"p__{s.id}" not in data.files for s in sequences) or any(f"c__{s.id}" not in data.files for s in natives):
            raise SystemExit(f"{path} does not hold every sequence it should")
        return {key: data[key] for key in data.files}


def _probabilities(held: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict[str, np.ndarray]:
    return {s.id: held[f"p__{s.id}"] for s in sequences}


def _ninth_files(folder: Path) -> dict[str, np.ndarray]:
    out = {}
    for path in sorted(folder.glob("ninth_*.npz")):
        if path.name.endswith(".partial.npz"):  # left by a run that was cut off while it wrote
            continue
        with np.load(path) as data:
            out[path.stem[len("ninth_"):]] = data["rows"]
    return out


def report(folder: Path, stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], stage2: Path | None = None,
           draws: int = DRAWS, seed: int = SEED) -> dict:
    """Every condition against the unchanged frames on the selection half. The published system is decoded with its
    published setting. With `stage2`, the stage-2 ensemble is scored beside it, with the setting that the project's
    rule gives it on the calibration half, and the published system minus stage 2 is given for every condition."""
    calibration, selection = halves["calibration"], halves["selection"]
    natives = of_kind(selection, "native")
    members = tuple(range(next(iter(stored.values())).shape[1]))
    folders = {PUBLISHED: folder, **({STAGE2: stage2} if stage2 is not None else {})}
    for system, where in folders.items():  # a folder must hold the system it is given as
        named = json.loads((where / "system.json").read_text()).get("system") if (where / "system.json").is_file() else None
        if named != system:
            raise SystemExit(f"{where} was written for '{named}', and is given as '{system}'")
    held = {system: {name: _held(where, name, selection) for name in CONDITIONS} for system, where in folders.items()}
    published = member_probabilities(dict(stored), selection, members)
    apart = max(float(np.abs(held[PUBLISHED][CLEAN][f"p__{s.id}"] - published[s.id]).max()) for s in selection)
    if apart > CLOSE:
        raise SystemExit(f"the unchanged frames, run again, do not give the stored outputs: {apart:.5f} apart")
    settings = {PUBLISHED: {"status": "published", "params": dict(B3)}}
    # what `classify` returns for the 521 single recordings from the benchmark's own frames, calibration half first
    known = {PUBLISHED: np.stack([np.append((c := classification(stored[s.id])).probabilities, float(c.seeds_agree))
                                  for half in benchmark.HALVES for s in of_kind(halves[half], "native")])}
    if stage2 is not None:
        on_calibration = _held(stage2, None, calibration)
        chosen = comparison.choose(_probabilities(on_calibration, calibration), calibration, drop_floors=True)
        settings[STAGE2] = {"status": chosen["status"], "floors_dropped": chosen["floors_dropped"], "params": chosen["params"],
                            "calibration": chosen["calibration"]}
        known[STAGE2] = np.stack([on_calibration[f"c__{s.id}"] for s in of_kind(calibration, "native")]
                                 + [held[STAGE2][CLEAN][f"c__{s.id}"] for s in natives])
    rows = [Row(f"{system} | {name}", ROUTER, "D", comparison.decoded(_probabilities(held[system][name], selection), settings[system]["params"]))
            for system in folders for name in CONDITIONS]
    scored = comparison.score_rows(rows, selection, draws, seed)
    clean = ablations.numbers(scored[f"{PUBLISHED} | {CLEAN}"])
    if round(clean["multi"]["segment_f1_50"]["value"], 4) != comparison.PUBLISHED_SELECTION_F1:
        raise SystemExit("the unchanged frames, run again, do not give the published number of the selection half")
    result: dict = {"benchmark": "development", "half": "selection", "sequences": len(selection), "draws": draws, "seed": seed,
                    "unchanged_against_stored": apart, "systems": {}}
    for system, where in folders.items():
        conditions = {}
        for name in CONDITIONS:
            mine = scored[f"{system} | {name}"]
            conditions[name] = {"kind": "unchanged" if name == CLEAN else "time" if name in TIME else "image",
                                "selection": ablations.numbers(mine), "classify": _classified(held[system][name], natives)}
            if name != CLEAN:
                conditions[name]["minus_unchanged"] = ablations.minus(mine, scored[f"{system} | {CLEAN}"], DESCRIBED)
        entry = {"setting": settings[system], "conditions": conditions}
        ninth = _ninth_files(where)
        if ninth:
            if not (where / KNOWN_FILE).is_file():
                raise SystemExit(f"{where} holds recordings of the ninth view and lacks {KNOWN_FILE}")
            with np.load(where / KNOWN_FILE) as data:
                through_reader = data["rows"]
            if through_reader.shape != known[system].shape:
                raise SystemExit(f"{where / KNOWN_FILE} does not hold the single recordings of both halves")
            entry["ninth_view"] = {name: ninth_part(rows_of_view, through_reader, draws, seed) for name, rows_of_view in ninth.items()}
            entry["reader_against_benchmark_frames"] = {
                "recordings": len(through_reader), "largest_difference_in_a_probability": float(np.abs(through_reader[:, :N] - known[system][:, :N]).max()),
                "recordings_given_another_view": int((through_reader[:, :N].argmax(1) != known[system][:, :N].argmax(1)).sum())}
        entry["system"] = json.loads((where / "system.json").read_text())
        result["systems"][system] = entry
    if stage2 is not None:
        result["published_minus_stage2"] = {name: ablations.minus(scored[f"{PUBLISHED} | {name}"], scored[f"{STAGE2} | {name}"], DESCRIBED)
                                            for name in CONDITIONS}
    return result


# ---------------------------------------------------------------------------------------------------- commands
def _lists(given: list[str]) -> dict[str, Path]:
    out = {}
    for text in given:
        name, _, path = text.partition("=")
        if not re.fullmatch(r"[a-z0-9]+", name) or not path:
            raise SystemExit("--ninth takes NAME=LIST, with a name of lower-case letters and digits")
        out[name] = Path(path)
    return out


def run_command(args: argparse.Namespace) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # plain float32, as in the published numbers
    torch.backends.cudnn.allow_tf32 = False
    import importlib.util

    if importlib.util.find_spec("scipy") is None:  # the blur needs it; say so before the long part
        raise SystemExit('the conditions need SciPy: pip install "echo-view-routing[evaluation]" (or pip install scipy)')
    lists = _lists(args.ninth)
    halves = ablations.development_halves()
    selection = halves["selection"]
    if args.system == "stage2":
        system, name = stage2_system(args.weights, args.device, args.batch), STAGE2
    else:
        system, name = followup.reference_system(args.weights, args.device, args.batch), PUBLISHED
    wanted = (*selection, *halves["calibration"]) if name == STAGE2 else selection
    table = ev9v.FrameTable.build(wanted, args.images, workers=args.workers, cache=args.cache)
    args.folder.mkdir(parents=True, exist_ok=True)
    if (args.folder / "system.json").is_file() and json.loads((args.folder / "system.json").read_text()).get("system") != name:
        raise SystemExit(f"{args.folder} holds the outputs of another system; give every system a folder of its own")
    (args.folder / "system.json").write_text(json.dumps({
        "system": name, "version": system.version, "models": len(system.models), "device": str(system.device), "torch": torch.__version__,
        "parameters_per_model": [int(sum(p.numel() for p in model.parameters())) for model in system.models]}) + "\n")
    if name == STAGE2 and not (args.folder / CALIBRATION_FILE).is_file():
        _save(run_condition(system, halves["calibration"], table, CLEAN), args.folder / CALIBRATION_FILE)
        print("done: the unchanged calibration half", flush=True)
    for condition in CONDITIONS:
        path = args.folder / f"{slug(condition)}.npz"
        if path.is_file():
            continue
        _save(run_condition(system, selection, table, condition), path)
        print(f"done: {condition}", flush=True)
    for label, listed in lists.items():
        path = args.folder / f"ninth_{label}.npz"
        if path.is_file():
            continue
        names = [line.strip() for line in listed.read_text().splitlines() if line.strip()]
        _save({"rows": read_ninth(system, args.images, names)}, path)
        print(f"done: the ninth view, list {label}, {len(names)} recordings", flush=True)
    if lists and not (args.folder / KNOWN_FILE).is_file():
        _save({"rows": read_known(system, args.images, halves)}, args.folder / KNOWN_FILE)
        print("done: the single recordings of both halves through the same reader", flush=True)


def report_command(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    halves = ablations.development_halves()
    stored = comparison._logits(args.logits, tuple(s for half in benchmark.HALVES for s in halves[half]))
    ablations.held_to_published(stored, halves["selection"])
    result = {"sha256": {"logits": comparison.file_sha256(args.logits)}, **report(args.folder, stored, halves, args.stage2, args.draws, SEED)}
    comparison._write(result, args.out)
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.stress", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="the models on the selection half under every condition; one file per condition")
    p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V")
    p.add_argument("--folder", type=Path, required=True, help="the work folder; a condition whose file exists is not computed again")
    p.add_argument("--system", choices=("published", "stage2"), default="published", help="the published system, or the stage-2 models as one ensemble")
    p.add_argument("--ninth", action="append", default=[], help="NAME=LIST: a text file with folder names of recordings of the view the system does not know")
    p.add_argument("--weights", type=Path, default=None, help="the folder of the published weights; downloaded when omitted")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="threads that decode frames")
    p.add_argument("--cache", type=Path, default=None, help="a folder that keeps the decoded frames for the next run")
    p.set_defaults(run=run_command)
    p = commands.add_parser("report", help="every condition scored against the unchanged frames; no model runs")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    p.add_argument("--folder", type=Path, required=True, help="the work folder of the run command for the published system")
    p.add_argument("--stage2", type=Path, default=None, help="the work folder of the run command with --system stage2")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=report_command)
    args = top.parse_args(argv)
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
