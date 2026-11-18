"""`echoview-train`: one command for the three training stages.

A recipe file names its training loop. `plan` is the loop of stages 1 and 2 (staged.py): update by update from a
sampling plan, frames decoded from the JPEG files, plain float32. `fast` is the loop of stage 3 (train.py): random
draws from the frame cache with input augmentation on the GPU. Each loop has its own options; `--help` after
`--config <recipe>` lists the ones of that recipe's loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOOPS = ("plan", "fast")


def loop_of(argv: list[str]) -> str:
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default="")
    known, _ = first.parse_known_args(argv)
    if not known.config:
        return "fast"
    try:
        loop = json.loads(Path(known.config).read_text()).get("loop", "fast")
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{known.config}: the recipe file cannot be read: {error}") from None
    if loop not in LOOPS:
        raise SystemExit(f"{known.config}: the loop must be one of {', '.join(LOOPS)}, not {loop!r}")
    return loop


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if loop_of(argv) == "plan":
        from .staged import main as run
    else:
        from .train import main as run
    return int(run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
