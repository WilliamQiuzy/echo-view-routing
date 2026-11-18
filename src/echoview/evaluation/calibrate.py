"""Choosing the decoder setting on the calibration half: the project's rule.

The grid is five half windows by eight thresholds. A threshold is a quantile of the decoder's scores inside the
native recordings of the calibration half, where no view change exists, so "0.999" means: one sample in a thousand
of a single-view recording scores this high. A setting is feasible when the recording-level macro-F1 stays at least
0.980 and the boundary recall on multi streams at least 0.95. Among the feasible settings the highest segment F1@50
on multi streams wins; an equal score goes to the earlier grid position. When nothing is feasible the rule says so and
returns the diagnostic setting (half window 9, quantile 0.95) so that numbers can still be shown.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .. import decoder
from . import metrics
from .benchmark import Sequence, of_kind

HALF_WINDOWS = (5, 7, 9, 11, 15)
QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999)
NMS_RADIUS = 5
NATIVE_FLOOR = 0.980
RECALL_FLOOR = 0.95
DIAGNOSTIC = (9, 0.95)
REPORTED = ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")


def grid(native_probabilities: list[np.ndarray]) -> list[dict]:
    """The settings in grid order; thresholds are quantiles of the scores inside the given native recordings."""
    out = []
    for half_window in HALF_WINDOWS:
        scores = np.concatenate([decoder.boundary_scores(p, half_window)[1:] for p in native_probabilities])
        for quantile in QUANTILES:
            out.append({"half_window": int(half_window), "threshold": float(np.quantile(scores, float(quantile))),
                        "nms_radius": NMS_RADIUS, "threshold_quantile": float(quantile)})
    return out


def decode_all(setting: Mapping, probabilities: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Every stream decoded at one setting. `scores` may hold the streams' boundary scores at the setting's half window."""
    half_window, threshold, radius = int(setting["half_window"]), float(setting["threshold"]), int(setting["nms_radius"])
    return {key: decoder.decode(p, half_window, threshold, radius, None if scores is None else scores[key]) for key, p in probabilities.items()}


def select_setting(records: list[dict], native_floor: float = NATIVE_FLOOR, recall_floor: float = RECALL_FLOOR) -> dict:
    """The rule on evaluated settings: feasible first, then the highest segment F1@50, then the earliest in the grid."""
    feasible = [r for r in records if r["native_macro_f1"] >= native_floor and (r["boundary_recall_multi"] or 0.0) >= recall_floor]
    if not feasible:
        return {"status": "infeasible", "chosen": None, "n_feasible": 0, "n_settings": len(records)}
    best = max(r["segment_f1_50_multi"] for r in feasible)
    chosen = min((r for r in feasible if r["segment_f1_50_multi"] == best), key=lambda r: r["order"])
    return {"status": "feasible", "chosen": chosen, "n_feasible": len(feasible), "n_settings": len(records)}


def choose(probabilities: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict:
    """The decoder setting for one system from its per-sample probabilities on the calibration half."""
    settings = grid([probabilities[s.id] for s in of_kind(sequences, "native")])
    records = []
    scored_window, scores = None, {}
    for order, setting in enumerate(settings):
        if setting["half_window"] != scored_window:  # the eight thresholds of one half window share its scores
            scored_window = setting["half_window"]
            scores = {key: decoder.boundary_scores(p, scored_window) for key, p in probabilities.items()}
        result = metrics.evaluate(decode_all(setting, probabilities, scores), sequences)
        records.append({"order": order, "params": setting, **{key: result[key] for key in REPORTED}})
    chosen = select_setting(records)
    if chosen["chosen"] is None:
        fallback = next(r for r in records if r["params"]["half_window"] == DIAGNOSTIC[0]
                        and abs(r["params"]["threshold_quantile"] - DIAGNOSTIC[1]) < 1e-9)
        best = max(records, key=lambda r: (r["segment_f1_50_multi"], -r["order"]))
        return {"status": "infeasible", "params": fallback["params"], "n_feasible": 0, "best_unconstrained": best}
    return {"status": "feasible", "params": chosen["chosen"]["params"], "n_feasible": chosen["n_feasible"]}
