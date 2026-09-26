"""Paired bootstrap over independent evaluation units (proposal §7.3).

Works on the per-fragment count tables the ladder writes (runs/<run>/fragments/<method>.<bank>.csv). The unit is the
stream for banks that use each cine at most once (pairs) and the cine for banks that reuse cines across streams
(multi), so recurring cines are never treated as independent. Every method is resampled with the same unit draws, so
differences between methods are paired. Without patient ids the intervals remain descriptive cine-level estimates.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASE_COLS = ["n", "t_true", "t_matched", "p_pred", "same_join", "false_split", "missed"]


def route_keys(df: pd.DataFrame) -> list[str]:
    """Threshold names present in a fragment table (columns acc|<key> / wrong|<key>)."""
    return sorted(c.split("|", 1)[1] for c in df.columns if c.startswith("acc|"))


def count_cols(df: pd.DataFrame) -> list[str]:
    return BASE_COLS + [f"{p}|{k}" for k in route_keys(df) for p in ("acc", "wrong")]


def unit_sums(df: pd.DataFrame, unit: str, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sum `cols` per unit ("stream" or "video_id"). Returns (sorted unit ids, n_units x n_cols matrix)."""
    g = df.groupby(unit, sort=True)[cols].sum()
    return g.index.to_numpy(), g.to_numpy(dtype=float)


def metrics(s: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    """Endpoints from summed counts (last axis = cols). NaN where a denominator is zero."""
    c = {k: s[..., i] for i, k in enumerate(cols)}
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"boundary_f1": 2 * c["t_matched"] / (c["p_pred"] + c["t_true"]),
               "false_split_rate": c["false_split"] / c["same_join"],
               "missed_rate": c["missed"] / c["t_true"]}
        for k in sorted({x.split("|", 1)[1] for x in cols if x.startswith("acc|")}):
            out[f"coverage|{k}"] = c[f"acc|{k}"] / c["n"]
            out[f"risk|{k}"] = c[f"wrong|{k}"] / c[f"acc|{k}"]
    return out


def resample_weights(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_units) multiplicities of units drawn with replacement; shared by all methods (paired)."""
    draws = rng.integers(0, n_units, size=(n_boot, n_units))
    return np.stack([np.bincount(d, minlength=n_units) for d in draws]).astype(float)


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else x


def _ci(rep: np.ndarray) -> list[float | None]:
    rep = rep[~np.isnan(rep)]
    return [None, None] if rep.size == 0 else [_num(v) for v in np.percentile(rep, [2.5, 97.5])]


def summarise(tables: dict[str, np.ndarray], cols: list[str], weights: np.ndarray, reference: str | None = None) -> dict:
    """Point estimate and 95% percentile interval for every endpoint and method, plus the paired difference to
    `reference` (same resampled units) for every other method."""
    reps = {m: metrics(weights @ x, cols) for m, x in tables.items()}
    points = {m: metrics(x.sum(0), cols) for m, x in tables.items()}
    out = {}
    for m in tables:
        res = {}
        for k, v in points[m].items():
            e = {"estimate": _num(v), "ci95": _ci(reps[m][k])}
            if reference in tables and m != reference:
                e["diff"] = _num(v - points[reference][k])
                e["diff_ci95"] = _ci(reps[m][k] - reps[reference][k])
            res[k] = e
        out[m] = res
    return out
