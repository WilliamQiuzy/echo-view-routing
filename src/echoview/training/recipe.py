"""Recipe files: the options of a training run as JSON.

A recipe's options are handed to the argument parser as if they had been typed before the command line's own, so they
are checked for type and allowed values in the same way, and an option on the command line wins.
"""
from __future__ import annotations

import json
from pathlib import Path

NOT_OPTIONS = ("description", "loop")  # what a recipe says about itself


def arguments(path: str, allowed: set[str]) -> list[str]:
    """The options of the recipe file as command-line arguments; an empty list without a file."""
    if not path:
        return []
    try:
        recipe = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: the recipe file cannot be read: {error}") from None
    if not isinstance(recipe, dict):
        raise SystemExit(f"{path}: a recipe is one JSON object of options")
    unknown = sorted(set(recipe) - allowed - set(NOT_OPTIONS))
    if unknown:
        raise SystemExit(f"{path}: unknown options {unknown}")
    out: list[str] = []
    for key, value in recipe.items():
        if key not in NOT_OPTIONS:
            out += [f"--{key.replace('_', '-')}", str(value)]
    return out
