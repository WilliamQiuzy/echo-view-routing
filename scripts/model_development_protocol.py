#!/usr/bin/env python
"""Fix the data protocol of model development round 1 before any candidate is trained (Task 1 of
docs/superpowers/plans/2026-10-02-model-development.md).

- The official EV9V train / validation / test split is kept; the test split is not used for selection or calibration.
- Validation sessions (recording-session prefixes of the file names) are split into a checkpoint-selection half and an
  operating-point-calibration half with a fixed seed, using only labels and session metadata. If a draw leaves a
  family out of either half, the seed is advanced and the attempt is recorded.
- Each half gets its own native, pair and multi-fragment banks built only from its own sessions, with recipe ids
  prefixed by the half, so no stream crosses the halves. Sample counts come from the frame counts at 10 Hz, so the
  protocol does not depend on any encoder.
- The training sources are the train-split clips and the existing 1,500-stream training bank.
Writes <run_dir>/protocol.json with clip and session lists, bank digests and source digests.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.banks import Bank, multi_bank, native_bank, pair_bank  # noqa: E402
from echo_routing.compose.recipe_builder import Pool, Recipe, make_multi_recipes, make_pair_recipes, recipe_to_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.sampling import stride_indices  # noqa: E402
from echo_routing.manifest import load_manifest, task_subset  # noqa: E402

HALVES = ("selection", "calibration")
N_CLASSES = 5


def session_of(video_id: str) -> str:
    return str(video_id).rsplit("_", 1)[0]


def sample_index(manifest: pd.DataFrame, hz: float) -> pd.DataFrame:
    """Encoder-independent index (video_id, split, raw_label, label_index, n_samples, session) at `hz`."""
    df = task_subset(manifest, "family5")
    df = df[df["frames_dir"].notna() & (df["n_frames"] > 0)].copy()
    df["n_samples"] = [len(stride_indices(int(n), float(f), hz)) for n, f in zip(df["n_frames"], df["fps_playback"])]
    df["label_index"] = df["family5_index"].astype(int)
    df["session"] = df["video_id"].map(session_of)
    return df[["video_id", "split", "raw_label", "label_index", "n_samples", "session"]].reset_index(drop=True)


def split_sessions(val: pd.DataFrame, seed: int, max_attempts: int) -> tuple[dict[str, list[str]], int]:
    """Random session-level halves that both contain every class; returns ({half: sessions}, attempts used)."""
    sessions = sorted(val["session"].unique())
    labels = val.groupby("session")["label_index"].apply(set).to_dict()
    for attempt in range(max_attempts):
        order = np.random.default_rng(seed + attempt).permutation(len(sessions))
        cut = len(sessions) // 2
        halves = {"selection": sorted(sessions[i] for i in order[:cut]), "calibration": sorted(sessions[i] for i in order[cut:])}
        if all(len(set().union(*(labels[s] for s in halves[h]))) == N_CLASSES for h in HALVES):
            return halves, attempt + 1
    raise SystemExit(f"no session split covering all {N_CLASSES} classes in {max_attempts} attempts")


def half_banks(idx_half: pd.DataFrame, half: str, cfg: dict, seed: int) -> dict[str, Bank]:
    """Native, pair and multi-fragment banks built only from one half's clips; recipe ids carry the half's name."""
    rc = cfg["recipes"]; mc = cfg.get("multi", {})
    pool = Pool.from_index(idx_half["video_id"], idx_half["label_index"], idx_half["n_samples"])
    pairs = make_pair_recipes(pool, int(rc["n_per_cell"]), np.random.default_rng(seed), edit_variant=rc["edit_variant"],
                              min_len=int(rc["min_len"]), max_len=int(rc["max_len"]), reuse=False)
    multi = make_multi_recipes(pool, int(mc.get("n_eval", 120)), np.random.default_rng(seed + 1000),
                               n_fragments=tuple(mc.get("n_fragments", (4, 8))), edit_variant=rc["edit_variant"],
                               edit_prob=float(mc.get("edit_prob", 0.5)), min_len=int(rc["min_len"]),
                               max_len=int(rc["max_len"]), reuse=True, p_same_next=float(mc.get("p_same_next", 0.35)))
    native = native_bank(idx_half.assign(split="validation"), "validation")
    return {"native": Bank(f"native_{half}", "validation", native.recipes),
            "pairs": Bank(f"pairs_{half}", "validation", tuple(Recipe(f"pairs_{half}_{r.recipe_id}", r.cell, r.fragments) for r in pairs)),
            "multi": Bank(f"multi_{half}", "validation", tuple(Recipe(f"multi_{half}_{r.recipe_id.split('_')[-1]}", r.cell, r.fragments) for r in multi))}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def bank_record(bank: Bank) -> dict:
    recipes = [recipe_to_dict(r) for r in bank.recipes]
    return {"bank_id": bank.bank_id, "n": len(recipes), "digest": digest(recipes), "recipes": recipes}


def build_protocol(cfg: dict, manifest: pd.DataFrame, manifest_sha: str) -> dict:
    pc = cfg["protocol"]; idx = sample_index(manifest, float(pc["hz"]))
    val = idx[idx["split"] == "validation"]
    halves, attempts = split_sessions(val, int(pc["seed"]), int(pc["max_attempts"]))
    out = {"protocol_version": pc["version"], "mapping_version": pc["mapping_version"], "hz": pc["hz"], "seed": pc["seed"],
           "source": {"manifest": "data/manifests/ev9v_native.csv", "manifest_sha256": manifest_sha},
           "split_rule": "validation sessions (file-name prefix) randomly halved with seed + attempt until both halves "
                         "contain all five families; labels and session metadata only, no model scores",
           "attempts": attempts, "test_split": "not used for selection or calibration",
           "validation": {}, "train": {}}
    for half in HALVES:
        sub = val[val["session"].isin(halves[half])]
        banks = half_banks(sub, half, cfg, int(pc["half_recipe_seed"][half]))
        out["validation"][half] = {
            "sessions": halves[half], "clips": sorted(sub["video_id"]), "n_clips": int(len(sub)),
            "class_counts": {int(k): int(v) for k, v in sub["label_index"].value_counts().sort_index().items()},
            "banks": {k: bank_record(b) for k, b in banks.items()}}
    tr = idx[idx["split"] == "train"]
    train_multi = multi_bank(idx, "train", cfg)
    out["train"] = {"clips": sorted(tr["video_id"]), "n_clips": int(len(tr)),
                    "class_counts": {int(k): int(v) for k, v in tr["label_index"].value_counts().sort_index().items()},
                    "multi_bank": {"bank_id": train_multi.bank_id, "n": len(train_multi.recipes),
                                   "digest": digest([recipe_to_dict(r) for r in train_multi.recipes])}}
    hist = {f"{k}_validation": f(idx, "validation", cfg) if k != "native" else native_bank(idx, "validation")
            for k, f in (("pairs", pair_bank), ("multi", multi_bank), ("native", None))}
    out["historical_validation_banks"] = {k: {"n": len(b.recipes), "digest": digest([recipe_to_dict(r) for r in b.recipes]),
                                              "use": "descriptive tables only"} for k, b in hist.items()}
    return out


def main() -> None:
    ap = base_parser("Model development protocol"); args = ap.parse_args()
    cfg = load_cfg(args); paths = load_paths()
    manifest = load_manifest(paths.manifest_path)
    manifest_sha = hashlib.sha256(Path(paths.manifest_path).read_bytes()).hexdigest()
    proto = build_protocol(cfg, manifest, manifest_sha)
    run_dir = new_run_dir(cfg, "model_development_protocol")
    (run_dir / "protocol.json").write_text(json.dumps(proto, indent=1))
    summary = {h: {"sessions": len(proto["validation"][h]["sessions"]), "clips": proto["validation"][h]["n_clips"],
                   "class_counts": proto["validation"][h]["class_counts"],
                   "banks": {k: (b["n"], b["digest"][:12]) for k, b in proto["validation"][h]["banks"].items()}} for h in HALVES}
    print(json.dumps({"run_dir": str(run_dir), "attempts": proto["attempts"], "halves": summary,
                      "train_multi": (proto["train"]["multi_bank"]["n"], proto["train"]["multi_bank"]["digest"][:16])}, indent=1))


if __name__ == "__main__":
    main()
