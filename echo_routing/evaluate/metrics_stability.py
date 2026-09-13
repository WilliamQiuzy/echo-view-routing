"""Stability proxy: fragments per minute within source-labelled single-view cines."""
from __future__ import annotations

import numpy as np


def fragments_per_minute(labels: np.ndarray, hz: float) -> float:
    labels = np.asarray(labels)
    if labels.size < 2 or hz <= 0:
        return 0.0
    changes = int(np.count_nonzero(np.diff(labels)))
    minutes = labels.size / hz / 60.0
    return changes / minutes
