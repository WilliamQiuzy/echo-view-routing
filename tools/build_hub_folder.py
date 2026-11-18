#!/usr/bin/env python3
"""Builds the folder that is uploaded to the Hugging Face Hub from the frozen training outputs.

  python tools/build_hub_folder.py --frozen <folder with C2_s0, C2_s1, C2_s2> --examples <folder with the sample clips> \
      --out <new folder> [--card MODEL_CARD.md]

For every seed the frozen `final.pt` is checked against its published checksum, loaded strictly into the network,
written as `model.safetensors` and read back: every tensor must be bit for bit the one of `final.pt`. The frozen
files are copied beside it unchanged, and the sample clips of the self-check go to `examples/`. Nothing is trained or
changed here. The script prints the checksums of the safetensors files; they belong in `src/echoview/reference.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from echoview import weights
from echoview.model import CLASSES, MEAN, STD, WindowModel, read_state
from echoview.reference import B3, SEEDS, VERSION, WEIGHT_SHA256

LOG = logging.getLogger("build_hub_folder")
COPIED = ("final.pt", "config.json", "decoder.json", "train_log.jsonl")
ENSEMBLE_DECODER = {
    "system": f"{VERSION}: the three C2 seeds as one probability ensemble",
    "input": "centre crop (short side 256, central 224 x 224) in grey levels, 10 Hz",
    "decoder": "B3 change-point decoder on the per-sample mean probability of the three seeds",
    "b3": {**B3, "threshold_quantile": 0.999, "temperature": 1.0},
    "b3_source": "the 0.999 quantile of the decoder's scores inside the native recordings of the EV9V calibration half",
}
SYSTEM = {
    "system": VERSION,
    "task": "echocardiography view recognition and view segmentation",
    "classes": list(CLASSES),
    "members": [weights.seed_folder(seed) for seed in SEEDS],
    "ensemble": "mean of the members' probabilities",
    "architecture": "torchvision mvit_v2_s video encoder, 512-d projection, a second input stem for the target-frame marker, linear head",
    "parameters_per_member": 34_600_000,
    "input": {"frame_rate_hz": 10, "frames_per_window": 16, "window_stride_samples": 5, "size": [224, 224],
              "preprocessing": "short side to 256 (bilinear), central 224 x 224 crop, mean of the three channels in each channel",
              "mean_0_255": list(MEAN), "std_0_255": list(STD)},
    "decoder": "see decoder.json",
    "code": "https://github.com/WilliamQiuzy/echo-view-routing",
}


def convert(source: Path, target: Path, seed: int) -> None:
    """final.pt -> model.safetensors, with every tensor compared after reading the new file back."""
    if weights.sha256(source) != WEIGHT_SHA256[seed]["final.pt"]:
        raise SystemExit(f"{source} is not the frozen file of seed {seed}: its checksum differs")
    state = read_state(source)
    WindowModel().load_state_dict(state, strict=True)
    # One metadata entry only: with several, their order in the file header changes from run to run and with it the
    # file's checksum. Where the file comes from is written in MANIFEST.json instead.
    save_file({key: value.contiguous() for key, value in state.items()}, str(target), metadata={"format": "pt"})
    back = load_file(str(target), device="cpu")
    if set(back) != set(state) or any(not torch.equal(back[key], state[key]) or back[key].dtype != state[key].dtype for key in state):
        raise SystemExit(f"{target} does not hold the tensors of {source}")


EXAMPLE_FILES = ("expected.json", "views.json")


def copy_examples(source: Path, target: Path) -> None:
    """The sample clips and their stored results; every clip must have a labelled view and a stored result."""
    clips = sorted(p for p in source.glob("*.mp4") if not p.name.startswith("."))
    views = json.loads((source / "views.json").read_text())
    expected = json.loads((source / "expected.json").read_text())
    if not clips or {p.name for p in clips} != set(views) or set(views) != set(expected["clips"]):
        raise SystemExit(f"{source}: the clips, views.json and expected.json do not name the same files")
    target.mkdir()
    for path in [*clips, *(source / name for name in EXAMPLE_FILES)]:
        shutil.copyfile(path, target / path.name)


def build(frozen: Path, examples: Path, out: Path, card: Path | None) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists already; name a new folder")
    out.mkdir(parents=True)
    copy_examples(examples, out / weights.EXAMPLES)
    for seed in SEEDS:
        source, target = frozen / weights.seed_folder(seed), out / weights.seed_folder(seed)
        target.mkdir()
        for name in COPIED:
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / "MANIFEST.json", target / "FROZEN_MANIFEST.json")
        convert(target / "final.pt", target / "model.safetensors", seed)
    (out / "decoder.json").write_text(json.dumps(ENSEMBLE_DECODER, indent=1) + "\n")
    (out / "config.json").write_text(json.dumps(SYSTEM, indent=1) + "\n")
    if card is not None:
        shutil.copyfile(card, out / "README.md")
    files = {str(path.relative_to(out)): {"sha256": weights.sha256(path), "bytes": path.stat().st_size}
             for path in sorted(out.rglob("*")) if path.is_file()}
    manifest = {"system": VERSION,
                "note": "model.safetensors holds the tensors of final.pt of the same folder, bit for bit; final.pt is the frozen training output",
                "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--card", type=Path, default=None)
    args = parser.parse_args()
    manifest = build(args.frozen, args.examples, args.out, args.card)
    for seed in SEEDS:
        name = f"{weights.seed_folder(seed)}/model.safetensors"
        LOG.info("%s  %s  %d bytes", manifest["files"][name]["sha256"], name, manifest["files"][name]["bytes"])
    LOG.info("%d files in %s", len(manifest["files"]) + 1, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
