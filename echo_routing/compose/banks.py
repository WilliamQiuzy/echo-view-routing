"""Deterministic recipe banks shared by every evaluation script, so all methods see identical streams.

Bank ids are stable: pairs_<split>, multi_<split>, native_<split>. Recipes are also written to
data/manifests/recipes/<bank>_<ckpt_hash>.json for the record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from echo_routing.compose.recipe_builder import Fragment, Pool, Recipe, make_multi_recipes, make_pair_recipes, recipe_to_dict
from echo_routing.compose.stream_assembler import Stream, render
from echo_routing.features.cache import VideoFeatures

SPLIT_SEED_OFFSET = {"train": 100, "validation": 0, "test": 1}


@dataclass(frozen=True)
class Bank:
    bank_id: str
    split: str
    recipes: tuple[Recipe, ...]


def _pool(idx: pd.DataFrame, split: str) -> Pool:
    sub = idx[idx["split"] == split]
    return Pool.from_index(sub["video_id"], sub["label_index"], sub["n_samples"])


def pair_bank(idx: pd.DataFrame, split: str, cfg: dict) -> Bank:
    rc = cfg["recipes"]
    rng = np.random.default_rng(int(rc["seed"]) + SPLIT_SEED_OFFSET[split])
    rec = make_pair_recipes(_pool(idx, split), int(rc["n_per_cell"]), rng, edit_variant=rc["edit_variant"],
                            min_len=int(rc["min_len"]), max_len=int(rc["max_len"]), reuse=(split == "train"))
    return Bank(f"pairs_{split}", split, tuple(rec))


def multi_bank(idx: pd.DataFrame, split: str, cfg: dict) -> Bank:
    rc = cfg["recipes"]; mc = cfg.get("multi", {})
    rng = np.random.default_rng(int(rc["seed"]) + 1000 + SPLIT_SEED_OFFSET[split])
    n = int(mc.get("n_train", 1500)) if split == "train" else int(mc.get("n_eval", 120))
    # Multi-fragment banks reuse cines across streams (never within a stream): 4-8 fragments per stream cannot be
    # served by the minority classes otherwise. They are a secondary setting; the pair banks stay the primary,
    # at-most-once evaluation units.
    rec = make_multi_recipes(_pool(idx, split), n, rng, n_fragments=tuple(mc.get("n_fragments", (4, 8))),
                             edit_variant=rc["edit_variant"], edit_prob=float(mc.get("edit_prob", 0.5)),
                             min_len=int(rc["min_len"]), max_len=int(rc["max_len"]), reuse=True,
                             p_same_next=float(mc.get("p_same_next", 0.35)))
    rec = [Recipe(f"multi_{split}_{r.recipe_id.split('_')[-1]}", r.cell, r.fragments) for r in rec]
    return Bank(f"multi_{split}", split, tuple(rec))


def native_bank(idx: pd.DataFrame, split: str) -> Bank:
    """Each native cine as a one-fragment stream whose id is the video id."""
    sub = idx[idx["split"] == split]
    rec = [Recipe(str(r.video_id), "native", (Fragment(str(r.video_id), int(r.label_index), "orig", 0, int(r.n_samples)),))
           for r in sub.itertuples(index=False)]
    return Bank(f"native_{split}", split, tuple(rec))


def render_bank(bank: Bank, loader: Callable[[str, str], VideoFeatures]) -> list[Stream]:
    return [render(r, loader) for r in bank.recipes]


def save_bank(bank: Bank, out_dir: Path, ckpt_hash: str) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bank.bank_id}_{ckpt_hash}.json"
    path.write_text(json.dumps([recipe_to_dict(r) for r in bank.recipes], indent=1))
    return path
