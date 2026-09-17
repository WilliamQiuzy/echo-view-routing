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

Decoder = Callable[[np.ndarray, np.ndarray | None, dict[str, Any]], tuple[np.ndarray, np.ndarray]]

METHOD_DESCRIPTIONS = {
    "b_file": "native-file mean probability, whole-file accept/defer",
    "b0": "per-frame argmax, no temporal processing",
    "b1": "probability moving average + hysteresis",
    "b2": "sticky HMM, Viterbi over posteriors",
    "b3": "JS divergence between left/right windows + persistent label change",
    "b4": "official MS-TCN (yabufarha/ms-tcn, pinned) on frozen features",
    "b4_reimpl": "our compact MS-TCN re-implementation (reference only, never reported)",
    "b5": "official ASFormer (ChinaYi/ASFormer, pinned) on frozen features",
    "b6": "official STFM (frozen seed100) applied with a sliding window",
    "b7": "official EchoViewCLIP stage-1 (frozen) applied with a sliding window",
    "b8": "EchoPrime released view classifier applied per frame",
}


def _b0(prob, feat, cfg, stream_id=None):
    return frame_argmax(prob), prob


def _b1(prob, feat, cfg, stream_id=None):
    c = cfg["decode"]["b1"]
    return smooth_labels(prob, int(c["window"]), int(c["min_run"]), str(c.get("mode", "prob"))), prob


def _b2(prob, feat, cfg, stream_id=None):
    return hmm_decode(prob, sticky_transitions(prob.shape[1], float(cfg["decode"]["b2"]["stay"]))), prob


def _b3(prob, feat, cfg, stream_id=None):
    c = cfg["decode"]["b3"]
    b = detect_boundaries(prob, int(c["half_window"]), float(c["threshold"]), int(c["nms_radius"]))
    return labels_from_boundaries(prob, b), prob


def _b4(prob, feat, cfg, stream_id=None):
    """Official MS-TCN predictions (cache/mstcn_official/<tag>/<stream_id>.npz produced by scripts/run_mstcn_official.py)."""
    from echo_routing.temporal.baselines.b4_mstcn_official import lookup_prob  # lazy
    p = lookup_prob(cfg, stream_id)
    return p.argmax(1), p


def _b4_reimpl(prob, feat, cfg, stream_id=None):
    """Our compact re-implementation; reference only, never reported."""
    from echo_routing.temporal.baselines.b4_mstcn import mstcn_decoder  # lazy: needs torch + checkpoint
    return mstcn_decoder(cfg)(prob, feat, cfg)


def _b5(prob, feat, cfg, stream_id=None):
    """Official ASFormer predictions (cache/asformer_official/<tag>/<stream_id>.npz from scripts/run_asformer_official.py)."""
    from echo_routing.temporal.baselines.b4_mstcn_official import lookup_prob  # same on-disk contract
    p = lookup_prob({"mstcn_official": {"pred_dir": (cfg.get("asformer_official") or {}).get("pred_dir")}}, stream_id)
    return p.argmax(1), p


def _lookup_factory(cfg_key: str):
    def _dec(prob, feat, cfg, stream_id=None):
        from echo_routing.temporal.baselines.b4_mstcn_official import lookup_prob
        p = lookup_prob({"mstcn_official": {"pred_dir": (cfg.get(cfg_key) or {}).get("pred_dir")}}, stream_id)
        return p.argmax(1), p
    return _dec


DECODERS: dict[str, Decoder] = {"b0": _b0, "b1": _b1, "b2": _b2, "b3": _b3, "b4": _b4, "b4_reimpl": _b4_reimpl, "b5": _b5,
                                "b6": _lookup_factory("stfm_windowed"), "b7": _lookup_factory("echoviewclip_windowed"),
                                "b8": _lookup_factory("echoprime_windowed")}


def register(method: str, fn: Decoder) -> None:
    DECODERS[method] = fn


def decode(method: str, prob: np.ndarray, cfg: dict[str, Any], feat: np.ndarray | None = None, stream_id: str | None = None) -> np.ndarray:
    """Labels only (encoder probabilities are used for scoring)."""
    return decode_with_prob(method, prob, cfg, feat, stream_id)[0]


def decode_with_prob(method: str, prob: np.ndarray, cfg: dict[str, Any], feat: np.ndarray | None = None,
                     stream_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(labels, probabilities to score segments with). Learned temporal models return their own posteriors."""
    if method not in DECODERS:
        raise KeyError(f"unknown decoder {method!r}; available: {sorted(DECODERS)}")
    prob = np.asarray(prob, dtype=float)
    labels, score_prob = DECODERS[method](prob, feat, cfg, stream_id)
    if labels.shape != (prob.shape[0],) or score_prob.shape != prob.shape:
        raise RuntimeError(f"decoder {method} returned shapes {labels.shape}/{score_prob.shape}, expected ({prob.shape[0]},)/{prob.shape}")
    return labels, np.asarray(score_prob, dtype=float)
