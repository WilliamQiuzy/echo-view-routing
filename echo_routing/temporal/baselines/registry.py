"""Uniform decoder interface: method id + per-sample probabilities -> per-sample labels.

Every decoder sees only the probability sequence (never labels, joins, or file names).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from echo_routing.temporal.baselines.b0_frame_argmax import frame_argmax
from echo_routing.temporal.baselines.b1_smoothing import smooth_labels
from echo_routing.temporal.baselines.b2_hmm import hmm_decode, sticky_transitions
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries

Decoder = Callable[[np.ndarray, dict[str, Any]], np.ndarray]

METHOD_DESCRIPTIONS = {
    "b_file": "native-file mean probability, whole-file accept/defer",
    "b0": "per-frame argmax, no temporal processing",
    "b1": "probability moving average + hysteresis",
    "b2": "sticky HMM, Viterbi over posteriors",
    "b3": "JS divergence between left/right windows + persistent label change",
    "b4": "MS-TCN on frozen features (learned temporal segmentation)",
    "b6": "STFM official video classifier (adapter)",
}


def _b0(prob, cfg):
    return frame_argmax(prob)


def _b1(prob, cfg):
    c = cfg["decode"]["b1"]
    return smooth_labels(prob, int(c["window"]), int(c["min_run"]), str(c.get("mode", "prob")))


def _b2(prob, cfg):
    return hmm_decode(prob, sticky_transitions(prob.shape[1], float(cfg["decode"]["b2"]["stay"])))


def _b3(prob, cfg):
    c = cfg["decode"]["b3"]
    b = detect_boundaries(prob, int(c["half_window"]), float(c["threshold"]), int(c["nms_radius"]))
    return labels_from_boundaries(prob, b)


DECODERS: dict[str, Decoder] = {"b0": _b0, "b1": _b1, "b2": _b2, "b3": _b3}


def register(method: str, fn: Decoder) -> None:
    DECODERS[method] = fn


def decode(method: str, prob: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    if method not in DECODERS:
        raise KeyError(f"unknown decoder {method!r}; available: {sorted(DECODERS)}")
    labels = DECODERS[method](np.asarray(prob, dtype=float), cfg)
    if labels.shape != (prob.shape[0],):
        raise RuntimeError(f"decoder {method} returned shape {labels.shape}, expected ({prob.shape[0]},)")
    return labels
