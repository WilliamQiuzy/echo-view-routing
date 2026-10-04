#!/usr/bin/env python
"""Contrast-only baseline for the tilt target (configs/experiments/geometry_pretraining_v1.yaml, pairs.controls).

How much of d_alpha can simple image statistics explain? A ridge regression (lambda 1, fixed in advance, not tuned)
maps [f1, f2, f2 - f1] to d_alpha. Here f is a slice's mean, SD, sharpness and relative sharpness. It is fitted on the
pairs of the pretraining people (the seed-0 bank) and scored on the held-out people's bank, twice: on the raw slices
before standardisation, and on the final input the model sees. The zero predictor (always d_alpha = 0) is the floor
reference. The same is done for |d_alpha|, since a cue may carry the size of a tilt but not its sign. To show where a
signal comes from, each feature is also fitted alone, and each feature's correlation with the slice's own tilt alpha
is given (held-out slices). Writes <out>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDA = 1.0
FEATURES = ("mean", "sd", "sharpness", "relative_sharpness")


def design(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f[:, 0], f[:, 1], f[:, 1] - f[:, 0]], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray) -> tuple:
    mu, sd = x.mean(0), x.std(0) + 1e-9; z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + LAMBDA * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return mu, sd, w, float(y.mean())


def predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, w, b = model
    return ((x - mu) / sd) @ w + b


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"mae_deg": float(np.abs(y - p).mean()), "r2": float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-bank", required=True); ap.add_argument("--heldout-bank", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); tb, hb = Path(a.train_bank), Path(a.heldout_bank)
    ty, hy = (pd.read_csv(b / "specs.csv")["d_alpha"].to_numpy(float) for b in (tb, hb))
    moving = pd.read_csv(hb / "specs.csv")["kind"].to_numpy() != "same_plane"; res = {"lambda": LAMBDA, "train_pairs": len(ty), "heldout_pairs": len(hy)}
    res["zero_predictor"] = {"all": scores(hy, np.zeros_like(hy)), "moving_plane": scores(hy[moving], np.zeros(moving.sum()))}
    hs = pd.read_csv(hb / "specs.csv"); alpha = np.stack([hs["alpha1_deg"], hs["alpha2_deg"]], 1).ravel()
    for name in ("raw", "final"):
        tf, hf = (np.load(b / f"features_{name}.npy").astype(np.float64) for b in (tb, hb))
        tx, hx = design(tf), design(hf)
        signed = fit_ridge(tx, ty); size = fit_ridge(tx, np.abs(ty)); p, q = predict(signed, hx), predict(size, hx)
        res[name] = {"signed": {"all": scores(hy, p), "moving_plane": scores(hy[moving], p[moving])},
                     "size": {"all": scores(np.abs(hy), q), "zero_predictor_all": scores(np.abs(hy), np.zeros_like(hy)),
                              "mean_predictor_all": scores(np.abs(hy), np.full_like(hy, np.abs(ty).mean()))},
                     "signed_by_feature": {f: scores(hy, predict(fit_ridge(design(tf[..., [j]]), ty), design(hf[..., [j]])))
                                           for j, f in enumerate(FEATURES)},
                     "feature_r_with_slice_alpha": {f: float(np.corrcoef(hf[..., j].ravel(), alpha)[0, 1]) for j, f in enumerate(FEATURES)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
