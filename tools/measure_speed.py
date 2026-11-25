#!/usr/bin/env python3
"""Measures how long the system works on a stream, for the published read-out and the two denser ones.

  python tools/measure_speed.py [--device cuda] [--minutes 1] [--repeats 5] [--batch 0] [--out speed.json]

The stream is the five sample clips of the self-check, joined and repeated to the asked length. Every window costs
the same forward pass whatever it shows, so the content of the stream does not change the time.

What is timed is `Ensemble.segment` on frames that are already decoded: the three seeds on every window, and the
decoder. Reading and decoding the files is not timed; it is the same work for every step. Every step is run once
before it is timed, because a first run also pays for what a machine sets up once. The summary of a step is the
median of its runs, with the fastest and the slowest beside it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import echoview
from echoview import selfcheck
from echoview.pipeline import HZ
from echoview.windows import STRIDE, window_plan

STEPS = (STRIDE, 2, 1)  # samples between two outputs of the models: the published step, then the two denser ones
TIMED = ("Ensemble.segment on decoded frames: the window model of every seed on every window, and the decoder. "
         "Reading and decoding the files is not timed. Every step is run once before its timed runs.")
RUNS = "the summary is a median: give an odd number of runs, three or more"


def stream_of(parts: list[np.ndarray], n_samples: int) -> np.ndarray:
    """The clips joined in their order and repeated until the stream has `n_samples` frames."""
    if n_samples < 1:
        raise ValueError("a stream has at least one sample")
    if not parts or sum(len(part) for part in parts) == 0:
        raise ValueError("the clips hold no frames")
    joined = np.concatenate(parts)
    return np.concatenate([joined] * -(-n_samples // len(joined)))[:n_samples]


def time_steps(system, frames: np.ndarray, repeats: int = 5, clock=time.perf_counter) -> list[dict]:
    """One row per read-out step: its windows, the seconds of every timed run and their summary."""
    if repeats < 3 or repeats % 2 == 0:
        raise ValueError(RUNS)
    minutes = len(frames) / HZ / 60.0
    rows = []
    for stride in STEPS:
        system.segment(frames, stride=stride)  # not timed
        seconds = []
        for _ in range(repeats):
            began = clock()
            system.segment(frames, stride=stride)
            seconds.append(clock() - began)
        median = float(np.median(seconds))
        rows.append({"read_every_s": stride / HZ, "samples_between_outputs": stride, "windows": len(window_plan(len(frames), stride)[0]),
                     "runs": repeats, "seconds": [float(s) for s in seconds], "seconds_median": median,
                     "seconds_min": float(min(seconds)), "seconds_max": float(max(seconds)),
                     "seconds_per_minute_of_stream": median / minutes, "times_real_time": 60.0 * minutes / median})
    published = rows[0]
    return [{**row, "times_the_published_step": row["seconds_median"] / published["seconds_median"],
             "windows_times_the_published_step": row["windows"] / published["windows"]} for row in rows]


def environment(system) -> dict:
    """The machine and the versions a time belongs to."""
    import torch

    device = str(system.device)
    name = torch.cuda.get_device_name(system.device) if device.startswith("cuda") else (platform.processor() or platform.machine())
    return {"device": device, "device_name": name, "batch": int(system.batch), "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "echoview": echoview.__version__}


def peak_gpu_memory_gib(system) -> float | None:
    """The most memory PyTorch has held on an NVIDIA GPU since the process began (weights, the stream's frames and the
    forward passes); None on any other device."""
    import torch

    if not str(system.device).startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(system.device) / 2 ** 30


def report(system, frames: np.ndarray, repeats: int = 5, clock=time.perf_counter) -> dict:
    steps = time_steps(system, frames, repeats, clock)
    return {"stream_seconds": len(frames) / HZ, "system": system.version, "reference_system": bool(system.is_reference), "timed": TIMED,
            "environment": environment(system), "steps": steps, "gpu_memory_peak_gib": peak_gpu_memory_gib(system)}


def write(content: dict, out: Path) -> None:
    """The file appears whole or not at all, and never in the place of an existing file."""
    if out.exists():
        raise FileExistsError(str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    try:
        partial.write_text(json.dumps(content, indent=1) + "\n")
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)


def table(content: dict) -> str:
    lines = [f"{content['system']} on {content['environment']['device']} ({content['environment']['device_name']}), "
             f"a stream of {content['stream_seconds']:.0f} s, batch {content['environment']['batch']}",
             "read every   windows   seconds, median (fastest to slowest)   per minute of stream   times real time   times the published step"]
    for row in content["steps"]:
        lines.append(f"{row['read_every_s']:8.1f} s {row['windows']:9d}   {row['seconds_median']:8.2f} ({row['seconds_min']:.2f} to {row['seconds_max']:.2f})"
                     f"{row['seconds_per_minute_of_stream']:25.2f} s {row['times_real_time']:17.1f} {row['times_the_published_step']:27.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="", help="a local folder with the weights and the sample clips; default: the published ones")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=0, help="windows per forward pass; default: the system's own (8 on a GPU, else 4)")
    parser.add_argument("--minutes", type=float, default=1.0, help="length of the stream")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs of every step (odd, three or more)")
    parser.add_argument("--out", default="", help="write the report here as JSON")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.minutes) and args.minutes > 0):
        parser.error("--minutes must be a number above zero")
    if args.out and Path(args.out).exists():
        parser.error(f"{args.out} exists; it is not replaced")
    if args.repeats < 3 or args.repeats % 2 == 0:
        parser.error(RUNS)
    if args.batch < 0:
        parser.error("--batch must not be negative")
    system = echoview.Ensemble(args.weights or None, device=args.device, batch=args.batch)
    clips = selfcheck.sample_clips(selfcheck.clips_folder("", system.weights_dir))
    try:
        parts = [echoview.read_recording(path).frames for path in clips]
    except echoview.ReadError as error:
        raise SystemExit(f"a sample clip could not be read: {error}") from None
    frames = stream_of(parts, max(1, round(args.minutes * 60 * HZ)))
    content = report(system, frames, args.repeats)
    print(table(content))
    if args.out:
        write(content, Path(args.out))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
