"""Native cine evaluation contracts; no labels enter a prediction window."""
from __future__ import annotations

import numpy as np

from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.windowed import window_centres

CLASSES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')


def sample_indices(n_frames: int, fps: float, hz: float = 10.0) -> np.ndarray:
    """Nearest decoded frame at each 10-Hz time not beyond the last frame."""
    if n_frames < 1 or not np.isfinite(fps) or fps <= 0 or hz <= 0:
        raise ValueError('positive frame count and finite positive source rate required')
    count = int(np.floor((n_frames - 1) * hz / fps + 1e-9)) + 1
    return np.minimum(np.rint(np.arange(count) * fps / hz).astype(np.int64), n_frames - 1)


def window_plan(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 1:
        raise ValueError('an empty sequence has no prediction windows')
    centres = window_centres(n_samples, 5)
    windows = np.stack([window_frames(int(c), n_samples, 16, 16) for c in centres])
    slots = np.array([np.flatnonzero(w == c)[len(np.flatnonzero(w == c)) // 2]
                      for c, w in zip(centres, windows)], np.int64)
    return centres, windows, slots


def preserve_unknown_map(raw_prob: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Retain an unsupported raw winner as an error instead of deleting it."""
    raw = np.asarray(raw_prob)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('finite matrix required')
    return np.array([mapping.get(int(k), -1) for k in raw.argmax(1)], np.int64)


def majority_label(labels: np.ndarray) -> int:
    lab = np.asarray(labels, np.int64)
    if not lab.size or ((lab < -1) | (lab > 4)).any():
        raise ValueError('nonempty five-family labels or explicit unknown required')
    k = int(np.bincount(np.where(lab == -1, 5, lab), minlength=6).argmax())
    return -1 if k == 5 else k


def native_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('empty evaluation cannot receive a score')
    cm = np.zeros((5, 6), np.int64)
    for r in rows:
        y, p = int(r['true']), int(r['pred'])
        if y not in range(5) or p not in range(-1, 5):
            raise ValueError('reference labels must be supported; unknown predictions remain errors')
        cm[y, 5 if p == -1 else p] += 1
    support = cm.sum(1)
    tp = cm[np.arange(5), np.arange(5)]
    denom = support + cm[:, :5].sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros(5, float), where=denom > 0)
    duration = sum(float(r['duration_s']) for r in rows)
    cuts = sum(int(r['cuts']) for r in rows)
    return {
        'n_clips': len(rows), 'accuracy': float(tp.sum() / len(rows)),
        'class_support': dict(zip(CLASSES, support.tolist())),
        'class_recall': {k: float(tp[i]/support[i]) if support[i] else None for i, k in enumerate(CLASSES)},
        'five_class_macro_f1': float(f1.mean()) if (support > 0).all() else None,
        'confusion': cm.tolist(), 'confusion_columns': [*CLASSES, 'unsupported'],
        'internal_cuts': cuts, 'internal_cuts_per_clip': cuts / len(rows),
        'duration_s': duration, 'internal_cuts_per_minute': cuts * 60 / duration if duration else None,
        'n_samples': sum(int(r['n_samples']) for r in rows),
    }
