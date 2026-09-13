"""B4: predictions of the OFFICIAL MS-TCN code, exported per stream by scripts/run_mstcn_official.py.

Layout: <pred_dir>/<stream_id>.npz with `prob` (T, C) = softmax of the last stage of models/<ds>/split_1/epoch-50.model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from echo_routing.errors import CacheMissError

_CACHE: dict[str, np.ndarray] = {}


def lookup_prob(cfg: dict, stream_id: str | None) -> np.ndarray:
    pred_dir = (cfg.get("mstcn_official") or {}).get("pred_dir")
    if not pred_dir:
        raise KeyError("b4 requires cfg['mstcn_official']['pred_dir'] (use --set mstcn_official.pred_dir=<dir>)")
    if stream_id is None:
        raise ValueError("b4 needs the stream id to look up official predictions")
    key = f"{pred_dir}/{stream_id}"
    if key not in _CACHE:
        path = Path(pred_dir) / f"{stream_id}.npz"
        if not path.is_file():
            raise CacheMissError(f"no official MS-TCN prediction for stream {stream_id!r} under {pred_dir}")
        with np.load(path) as z:
            _CACHE[key] = z["prob"].astype(float)
    return _CACHE[key]
