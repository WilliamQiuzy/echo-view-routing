"""B-file: native-file-level aggregation with whole-file accept / defer (mandatory comparator)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FileDecision:
    view: int
    score: float
    accepted: bool


def file_score(prob: np.ndarray, method: str = "mean") -> tuple[int, float]:
    """Aggregate a cine's per-frame probabilities to (view, score)."""
    prob = np.asarray(prob)
    if method == "mean":
        m = prob.mean(0); view = int(m.argmax()); return view, float(m[view])
    if method == "p10_agree":
        view = int(prob.mean(0).argmax())
        p10 = float(np.percentile(prob[:, view], 10)); agree = float((prob.argmax(1) == view).mean())
        return view, p10 * agree
    raise ValueError(f"unknown method {method!r}")


def decide(prob: np.ndarray, tau: float, method: str = "mean") -> FileDecision:
    view, score = file_score(prob, method)
    return FileDecision(view=view, score=score, accepted=score >= tau)
