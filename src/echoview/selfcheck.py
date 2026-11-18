"""The self-check: the weights' checksums, then five public sample clips against their stored results.

The clips are EV9V recordings (CC BY 4.0), one per view. They are published with the weights (`examples/` on the Hub),
not in the code repository. The check reads the first 3 s of each, classifies it, joins
the five pieces into one stream and segments it. It passes when every view and the stream's pieces are the stored
ones and no probability moved by more than the tolerance (video decoders differ slightly between machines).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import frames as fr
from . import weights as wt
from .pipeline import Ensemble

PROBABILITY_TOLERANCE = 0.03
SAMPLES_PER_CLIP = 30  # the first 3 s of every clip, so that the check also finishes on a laptop
BOUNDARY_TOLERANCE_S = 0.2


def clips_folder(given: str, weights_dir: Path | None, writing: bool = False) -> Path:
    """--clips when given, else `examples/` beside the weights, else a download from the Hub. The stored results must
    be there, unless this run writes them."""
    try:
        folder = Path(given) if given else wt.examples_folder(weights_dir)
    except wt.WeightsError as error:
        raise SystemExit(str(error)) from None
    if not writing and not (folder / "expected.json").is_file():
        raise SystemExit(f"{folder} holds no sample clips: expected.json is missing")
    return folder


def sample_clips(folder: Path) -> list[Path]:
    clips = sorted(p for p in folder.glob("*.mp4") if not p.name.startswith("."))
    if not clips:
        raise SystemExit(f"{folder} holds no sample clip (*.mp4)")
    return clips


def run_samples(system: Ensemble, folder: Path) -> dict:
    """The clips' views and the segmentation of their join, in the form of expected.json."""
    clips = sample_clips(folder)
    parts = [fr.read_recording(path).frames[:SAMPLES_PER_CLIP] for path in clips]
    out: dict = {"clips": {}, "stream": {}}
    for path, frames in zip(clips, parts):
        result = system.classify(frames)
        out["clips"][path.name] = {"view": result.view, "probability": result.probability,
                                   "probabilities": result.probabilities.tolist(), "samples_10hz": len(frames)}
    stream = system.segment(np.concatenate(parts))
    out["stream"] = {"order": [p.name for p in clips], "seconds": len(stream.labels) / fr.HZ,
                     "segments": [[start, end, view] for start, end, view, _ in stream.segments]}
    return out


def compare_clips(got: dict, expected: dict, truth: dict) -> list[str]:
    """Prints one line per clip and returns the names of those that differ from the stored result."""
    problems = []
    for name, want in expected["clips"].items():
        have = got["clips"].get(name)
        if have is None:
            problems.append(f"{name} is missing")
            continue
        gap = float(np.abs(np.array(have["probabilities"]) - np.array(want["probabilities"])).max())
        ok = have["view"] == want["view"] and gap <= PROBABILITY_TOLERANCE and have["samples_10hz"] == want["samples_10hz"]
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {have['view']} {have['probability']:.3f} (stored {want['view']} {want['probability']:.3f}, "
              f"labelled {truth[name]}; largest probability gap {gap:.4f})")
        if not ok:
            problems.append(name)
    return problems


def compare_stream(got: dict, expected: dict) -> list[str]:
    have_views = [view for _, _, view in got["stream"]["segments"]]
    want_views = [view for _, _, view in expected["stream"]["segments"]]
    shifts = [abs(a[0] - b[0]) for a, b in zip(got["stream"]["segments"], expected["stream"]["segments"])]
    ok = have_views == want_views and max(shifts, default=0.0) <= BOUNDARY_TOLERANCE_S
    print(f"{'ok  ' if ok else 'FAIL'} stream of the {len(expected['stream']['order'])} clips joined: {' > '.join(have_views)}")
    for start, end, view in got["stream"]["segments"]:
        print(f"       {start:6.1f} - {end:6.1f} s  {view}")
    return [] if ok else ["stream segmentation"]


def run(args: argparse.Namespace) -> int:
    from .cli import build

    system = build(args, verify=False)
    folder = clips_folder(args.clips, system.weights_dir, writing=args.write_expected)
    problems = []
    for name, ok in wt.verify(system.weights_dir, system.seeds).items():
        print(f"{'ok  ' if ok else 'FAIL'} checksum {name}")
        if not ok:
            problems.append(f"checksum of {name}")
    began = time.time()
    print(f"reading the first {SAMPLES_PER_CLIP / fr.HZ:.0f} s of each sample clip: seconds on a GPU, two to three minutes on a laptop CPU ...", flush=True)
    got = run_samples(system, folder)
    expected_path = folder / "expected.json"
    if args.write_expected:
        expected_path.write_text(json.dumps(got, indent=1) + "\n")
        print(f"wrote {expected_path}")
        return 0
    expected = json.loads(expected_path.read_text())
    truth = json.loads((folder / "views.json").read_text())
    problems += compare_clips(got, expected, truth)
    problems += compare_stream(got, expected)
    print(f"{time.time() - began:.0f} s on {system.device}")
    print("SELF-CHECK PASSED" if not problems else "SELF-CHECK FAILED: " + "; ".join(problems))
    return 0 if not problems else 1
