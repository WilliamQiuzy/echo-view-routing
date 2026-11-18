"""Echocardiography view recognition and view segmentation with the reference ensemble.

  echoview selfcheck
  echoview classify  INPUT [INPUT ...] [--out views.csv]
  echoview segment   INPUT [INPUT ...] [--out segments.csv] [--samples samples.csv]
  echoview weights
  echoview evaluate  --images EV9V/Images [--out report.json]
  echoview mimic-table --labels V4.csv --root MIMIC_ROOT --out ours.csv [--limit N]

INPUT is a DICOM cine, a video file, a folder of frame images, or a folder that holds any of these.
classify gives one view per recording. segment joins the inputs in the given order into one stream and cuts it into
views. weights downloads the published weights if needed and checks them. evaluate runs the EV9V development
evaluation. mimic-table writes a table with the columns of the MIMIC-IV-Echo view table, filled by this model.
The model knows five views (PLAX, PSAX, A4C, A5C, SC4C) and always answers one of them.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from pathlib import Path

import numpy as np

from . import frames as fr
from . import weights as wt
from .model import CLASSES
from .pipeline import Ensemble
from .reference import SEEDS

CLASSIFY_HEADER = ("file", "predicted_view", "view_probability", *(f"p_{c}" for c in CLASSES), "seeds_agree", "source_frames",
                   "fps", "samples_10hz", "windows", "reader", "view_classifier_version")
SEGMENT_HEADER = ("segment", "start_s", "end_s", "view", "mean_probability", "view_classifier_version")


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(s) for s in text.split(","))
    except ValueError:
        raise SystemExit("--seeds takes numbers separated by commas, for example 0,1,2") from None
    if not seeds or not set(seeds) <= set(SEEDS) or len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds names seeds among {', '.join(str(s) for s in SEEDS)}, each once")
    return seeds


def build(args: argparse.Namespace, verify: bool = True) -> Ensemble:
    """The system the command line asked for; a problem with the weights ends the command with its message."""
    if args.batch < 0:
        raise SystemExit("--batch must not be negative")
    try:
        system = Ensemble(args.weights or None, seeds=parse_seeds(args.seeds), device=args.device, batch=args.batch,
                          verify=verify and not args.unverified)
    except wt.WeightsError as error:
        raise SystemExit(str(error)) from None
    except RuntimeError as error:  # PyTorch does not know the device, or cannot use it
        raise SystemExit(f"--device {args.device}: {str(error).splitlines()[0]}") from None
    print(f"model {system.version} on {system.device}", file=sys.stderr)
    return system


def write_csv(path: Path, header: list[str] | tuple[str, ...], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def note_reader(path: Path, recording: fr.Recording) -> None:
    if recording.reader == fr.PYDICOM_READER:
        print(f"NOTE {path.name}: decoded by pydicom, not by the reader behind the MIMIC numbers", file=sys.stderr)


def recordings_of(inputs: list[str]) -> list[Path]:
    paths = fr.find_recordings(inputs)
    if not paths:
        raise SystemExit("no recording found in the inputs")
    return paths


# ------------------------------------------------------------------------------------------------ classify
def classify_row(path: Path, recording: fr.Recording, system: Ensemble) -> tuple[list, str]:
    result = system.classify(recording.frames)
    row = [str(path), result.view, f"{result.probability:.6f}", *[f"{p:.6f}" for p in result.probabilities],
           int(result.seeds_agree), recording.n_source_frames, f"{recording.fps:.4g}", len(recording.frames),
           result.n_windows, recording.reader, system.version]
    return row, f"{result.view:5s} {result.probability:.3f}  {path}"


def classify(args: argparse.Namespace) -> int:
    paths = recordings_of(args.inputs)
    system = build(args)
    done, failed = 0, 0
    with contextlib.ExitStack() as stack:
        handle, writer = None, None
        if args.out:  # rows are written as they are made, so an interrupted run keeps what it has done
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(Path(args.out).open("w", newline=""))
            writer = csv.writer(handle)
            writer.writerow(CLASSIFY_HEADER)
        for path in paths:
            try:
                recording = fr.read_recording(path, args.fps)
            except fr.ReadError as error:
                print(f"SKIPPED {error}", file=sys.stderr)
                failed += 1
                continue
            note_reader(path, recording)
            row, line = classify_row(path, recording, system)
            if writer is not None:
                writer.writerow(row)
                handle.flush()
            done += 1
            print(line)
    if args.out:
        print(f"wrote {args.out}: {done} recordings", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(paths)} inputs could not be read", file=sys.stderr)
    return 1 if failed and not done else 0


# ------------------------------------------------------------------------------------------------- segment
def segment(args: argparse.Namespace) -> int:
    paths = recordings_of(args.inputs)
    try:
        recordings = [fr.read_recording(path, args.fps) for path in paths]
    except fr.ReadError as error:
        raise SystemExit(f"an input could not be read, so no stream was built: {error}") from None
    for path, recording in zip(paths, recordings):
        note_reader(path, recording)
    system = build(args)
    stream = np.concatenate([r.frames for r in recordings])
    source = np.concatenate([np.full(len(r.frames), k) for k, r in enumerate(recordings)])
    result = system.segment(stream)
    rows = [[k, f"{start:.1f}", f"{end:.1f}", view, f"{probability:.4f}", system.version]
            for k, (start, end, view, probability) in enumerate(result.segments)]
    print(f"stream of {len(recordings)} recording(s), {len(stream) / fr.HZ:.1f} s")
    for _k, start, end, view, probability, _version in rows:
        print(f"  {start:>6s} - {end:>6s} s  {view:5s} {probability}")
    if args.out:
        write_csv(Path(args.out), SEGMENT_HEADER, rows)
    if args.samples:
        sample_rows = [[i, f"{i / fr.HZ:.1f}", str(paths[int(source[i])]), CLASSES[int(result.labels[i])],
                        *[f"{p:.5f}" for p in result.probabilities[i]]] for i in range(len(stream))]
        write_csv(Path(args.samples), ["sample", "time_s", "file", "view", *[f"p_{c}" for c in CLASSES]], sample_rows)
    return 0


# ------------------------------------------------------------------------------------------------- weights
def show_weights(args: argparse.Namespace) -> int:
    """Finds the weights (downloading them if no folder is given), checks every file and says where they are."""
    seeds = parse_seeds(args.seeds)
    try:
        folder = wt.resolve(args.weights or None, seeds)
        checks = wt.verify(folder, seeds)
    except wt.WeightsError as error:
        raise SystemExit(str(error)) from None
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} checksum {name}")
    print(f"weights folder: {folder}")
    return 0 if all(checks.values()) else 1


# ------------------------------------------------------------------------------------------------ evaluate
def evaluate(args: argparse.Namespace) -> int:
    """The EV9V development evaluation: each seed, the ensemble by the rule and the ensemble as published."""
    import importlib.util

    import torch

    from .evaluation import benchmark, ev9v, run

    if importlib.util.find_spec("scipy") is None:  # the stream scorecard needs it; say so before the long part
        raise SystemExit('the evaluation needs SciPy: pip install "echo-view-routing[evaluation]" (or pip install scipy)')
    # Plain float32 arithmetic on the GPU, as in the published numbers: TF32 changes logits in the fourth decimal.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    halves = benchmark.load()
    system = build(args)  # before the frames are decoded: a problem with the weights shows at once
    try:
        tables = {half: ev9v.FrameTable.build(halves[half], args.images, args.workers, args.cache or None) for half in benchmark.HALVES}
    except ev9v.DataError as error:
        raise SystemExit(str(error)) from None
    report = run.evaluate_reference(system, halves, tables)
    for line in run.headline(report):
        print(line)
    if args.out:
        run.write(report, args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


# -------------------------------------------------------------------------------------------------- parser
def parser() -> argparse.ArgumentParser:
    from . import mimic_table, selfcheck

    top = argparse.ArgumentParser(prog="echoview", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights", default="", help="folder with C2_s0, C2_s1, C2_s2 (default: $ECHOVIEW_WEIGHTS, else a download from the Hugging Face Hub)")
    common.add_argument("--unverified", action="store_true",
                        help="run weight files that are not the published ones, such as seeds you trained yourself")
    common.add_argument("--device", default="auto", help="auto (an NVIDIA GPU through CUDA if PyTorch sees one, else the CPU), cpu, cuda, cuda:1, mps")
    common.add_argument("--seeds", default="0,1,2", help="seeds to run; the reference system is all three")
    common.add_argument("--batch", type=int, default=0, help="windows per forward pass (0: 8 on a GPU, 4 on the CPU)")
    common.add_argument("--fps", type=float, default=None,
                        help="frame rate for inputs that carry none (an image folder always needs it); a file's own rate is never replaced")
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("selfcheck", parents=[common], help="run the sample clips and compare with the stored results")
    p.add_argument("--clips", default="", help="folder with the sample clips (default: examples/ beside the weights, else a download)")
    p.add_argument("--write-expected", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(run=selfcheck.run)
    p = commands.add_parser("classify", parents=[common], help="one view per recording")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="", help="CSV file to write")
    p.set_defaults(run=classify)
    p = commands.add_parser("segment", parents=[common], help="join the inputs into one stream and cut it into views")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="", help="CSV of the pieces")
    p.add_argument("--samples", default="", help="CSV with the view and probabilities of every 0.1 s sample")
    p.set_defaults(run=segment)
    p = commands.add_parser("weights", parents=[common], help="download the published weights if needed and check them")
    p.set_defaults(run=show_weights)
    p = commands.add_parser("evaluate", parents=[common], help="the EV9V development evaluation (needs the EV9V Images folder)")
    p.add_argument("--images", required=True, help="the Images folder of EV9V: Images/<recording>/<frame>.jpg")
    p.add_argument("--out", default="", help="JSON file for the full report")
    p.add_argument("--cache", default="", help="folder that keeps the decoded frames between runs (about 2 GB)")
    p.add_argument("--workers", type=int, default=8, help="threads that decode frames")
    p.set_defaults(run=evaluate)
    p = commands.add_parser("mimic-table", parents=[common], help="the MIMIC-IV-Echo view table filled by this model")
    p.add_argument("--labels", required=True, help="the existing view table (its first seven columns are copied)")
    p.add_argument("--root", required=True, help="folder that holds the table's dicom_file paths (files/p10/...)")
    p.add_argument("--out", required=True, help="CSV to write; an existing file is continued")
    p.add_argument("--limit", type=int, default=0, help="stop after this many new rows (0: no limit)")
    p.add_argument("--workers", type=int, default=4, help="threads that read files")
    p.set_defaults(run=mimic_table.run)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
