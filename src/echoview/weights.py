"""Finding the weights of the reference system and checking that they are the published files.

A weights folder holds one sub-folder per seed, `C2_s0`, `C2_s1` and `C2_s2`. Each holds `model.safetensors` or the
frozen training output `final.pt` (the first is used when both are there). The published folder also holds
`examples/`: five public EV9V clips and the stored results of the self-check. The folder is, in this order:

  1. the folder given by the caller (`--weights` on the command line);
  2. the folder named by the environment variable ECHOVIEW_WEIGHTS;
  3. a download of the published files from the Hugging Face Hub, kept in the Hub's own cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .reference import HUB_REPO, HUB_REVISION, SEEDS, WEIGHT_SHA256

ENVIRONMENT_VARIABLE = "ECHOVIEW_WEIGHTS"
FILE_NAMES = ("model.safetensors", "final.pt")  # in the order of preference
EXAMPLES = "examples"


class WeightsError(RuntimeError):
    """The weights are missing or are not the published files; the message says which and what to do."""


def seed_folder(seed: int) -> str:
    return f"C2_s{int(seed)}"


def seed_file(folder: str | Path, seed: int) -> Path:
    """The weight file of one seed inside a weights folder."""
    for name in FILE_NAMES:
        candidate = Path(folder) / seed_folder(seed) / name
        if candidate.is_file():
            return candidate
    raise WeightsError(f"{Path(folder) / seed_folder(seed)} holds neither {FILE_NAMES[0]} nor {FILE_NAMES[1]}")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_published(path: str | Path, seed: int) -> bool:
    """True when the file is byte for byte one of the published weight files of this seed."""
    expected = WEIGHT_SHA256.get(int(seed), {}).get(Path(path).name, "")
    return bool(expected) and sha256(path) == expected


def verify(folder: str | Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, bool]:
    """Every seed's weight file against its published checksum, by its path inside the folder."""
    out = {}
    for seed in seeds:
        path = seed_file(folder, seed)
        out[f"{seed_folder(seed)}/{path.name}"] = is_published(path, seed)
    return out


def _snapshot(patterns: list[str], repo: str, revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        raise WeightsError("downloading from the Hub needs the package huggingface_hub: pip install huggingface_hub") from error
    try:
        return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns))
    except HfHubHTTPError as error:
        raise WeightsError(
            f"{', '.join(patterns)} could not be downloaded from {repo}: {type(error).__name__}. The repository may be private: "
            f"log in with `hf auth login`, or point {ENVIRONMENT_VARIABLE} at a folder that holds the weights."
        ) from error


def download(repo: str = HUB_REPO, revision: str = HUB_REVISION, seeds: tuple[int, ...] = SEEDS) -> Path:
    """The published safetensors files from the Hugging Face Hub; returns the folder they are in."""
    return _snapshot([f"{seed_folder(seed)}/{FILE_NAMES[0]}" for seed in seeds], repo, revision)


def examples_folder(weights_dir: str | Path | None = None, repo: str = HUB_REPO, revision: str = HUB_REVISION) -> Path:
    """The sample clips of the self-check: `examples/` beside the weights, else a download from the Hub."""
    if weights_dir is not None and (Path(weights_dir) / EXAMPLES / "expected.json").is_file():
        return Path(weights_dir) / EXAMPLES
    return _snapshot([f"{EXAMPLES}/*"], repo, revision) / EXAMPLES


def resolve(folder: str | Path | None = None, seeds: tuple[int, ...] = SEEDS) -> Path:
    """The weights folder to use: the caller's, the environment's, or a download."""
    given = folder if folder is not None else os.environ.get(ENVIRONMENT_VARIABLE) or None
    if given is None:
        return download(seeds=seeds)
    path = Path(given).expanduser()
    if not path.is_dir():
        raise WeightsError(f"{path} is not a folder; it should hold {', '.join(seed_folder(s) for s in seeds)}")
    return path
