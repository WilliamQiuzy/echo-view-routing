#!/usr/bin/env python
"""A14 core (configs/experiments/video_local_router_v1.yaml): disagreement zones between Base and Alternative, their
application, the full-sequence segment utility of one zone, the fixed P/V features, and synthetic tests.

  python scripts/local_router.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

MERGE, CTX, C = 16, 16, 5
EPS = 1e-12


def zones(base: np.ndarray, alt: np.ndarray, merge: int = MERGE) -> list[tuple[int, int]]:
    """Half-open [a, b) zones: maximal disagreement runs, neighbours at most `merge` agreeing samples apart merged."""
    d = np.asarray(base) != np.asarray(alt); out = []; i, n = 0, d.size
    while i < n:
        if d[i]:
            j = i
            while j < n and d[j]:
                j += 1
            if out and i - out[-1][1] <= merge:
                out[-1] = (out[-1][0], j)
            else:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def apply(base: np.ndarray, alt: np.ndarray, zs: list[tuple[int, int]], accept) -> np.ndarray:
    out = np.asarray(base).copy()
    for (a, b), ok in zip(zs, accept):
        if ok:
            out[a:b] = np.asarray(alt)[a:b]
    return out


def seq_f1(pred: np.ndarray, ref: np.ndarray) -> dict:
    p, t = labels_to_segments(np.asarray(pred)), labels_to_segments(np.asarray(ref)); r = segment_f1(p, t, 0.5)
    return {"f1": float(r["f1"]), "tp": int(r["matched"]), "pred": len(p), "ref": len(t)}


def zone_utilities(base, alt, ref, zs) -> list[dict]:
    """Per zone: the sequence F1 with Base against Base with only this zone replaced; positive when strictly higher."""
    b0 = seq_f1(base, ref); out = []
    for z in zs:
        after = seq_f1(apply(base, alt, [z], [True]), ref)
        out.append({"f1_base": b0["f1"], "f1_after": after["f1"], "dtp": after["tp"] - b0["tp"], "dfp": (after["pred"] - after["tp"]) - (b0["pred"] - b0["tp"]),
                    "dfn": (after["ref"] - after["tp"]) - (b0["ref"] - b0["tp"]), "positive": bool(after["f1"] > b0["f1"]), "length": int(z[1] - z[0])})
    return out


def _runs(x: np.ndarray) -> int:
    return int(1 + (np.diff(x) != 0).sum()) if x.size else 0


def _major(x: np.ndarray) -> tuple[int, float]:
    c = np.bincount(np.asarray(x, int), minlength=C); return int(c.argmax()), float(c.max() / max(1, x.size))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q); kl = lambda a, b: float(np.sum(a * (np.log(a + EPS) - np.log(b + EPS))))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def onehot(k: int | None) -> np.ndarray:
    v = np.zeros(C)
    if k is not None:
        v[k] = 1.0
    return v


def features_p(prob: np.ndarray, base: np.ndarray, alt: np.ndarray, z: tuple[int, int]) -> np.ndarray:
    """The fixed P features of one zone (predictions and A08 probability statistics only)."""
    a, b = z; n = prob.shape[0]; la, rb = max(0, a - CTX), min(n, b + CTX)
    pz, pl, pr = prob[a:b].mean(0), (prob[la:a].mean(0) if a > la else np.zeros(C)), (prob[b:rb].mean(0) if rb > b else np.zeros(C))
    bm, bf = _major(base[a:b]); am, af = _major(alt[a:b]); left = int(base[a - 1]) if a > 0 else None; right = int(base[b]) if b < n else None
    top = np.sort(prob[a:b], axis=1); margin = float((top[:, -1] - top[:, -2]).mean()); ent = float(-(prob[a:b] * np.log(prob[a:b] + EPS)).sum(1).mean())
    f = [np.log1p(b - a), (a - la) / CTX, (rb - b) / CTX, _runs(base[a:b]), bf, _runs(alt[a:b]), af, float(left is not None), float(right is not None),
         float(am == left), float(am == right), float(bm == left), float(bm == right), margin, ent, (_js(pz, pl) if a > la else 0.0), (_js(pz, pr) if rb > b else 0.0)]
    return np.concatenate([np.asarray(f, float), onehot(bm), onehot(am), onehot(left), onehot(right), pz, pl, pr])


def features_v(prob: np.ndarray, emb: np.ndarray, base: np.ndarray, alt: np.ndarray, z: tuple[int, int]) -> np.ndarray:
    """P's features plus pooled unit A08 embeddings (per-sample, nearest centre): zone-left and right-zone differences,
    their norms and the two cosine distances; missing contexts give zeros."""
    a, b = z; n = emb.shape[0]; la, rb = max(0, a - CTX), min(n, b + CTX)
    u = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6); mz = u[a:b].mean(0)
    ml = u[la:a].mean(0) if a > la else None; mr = u[b:rb].mean(0) if rb > b else None
    d1 = mz - ml if ml is not None else np.zeros_like(mz); d2 = mr - mz if mr is not None else np.zeros_like(mz)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + EPS))  # noqa: E731
    extra = [np.linalg.norm(d1), np.linalg.norm(d2), (1 - cos(mz, ml)) if ml is not None else 0.0, (1 - cos(mz, mr)) if mr is not None else 0.0]
    return np.concatenate([features_p(prob, base, alt, z), d1, d2, np.asarray(extra, float)])


# ----------------------------------------------------------------------------------------------- synthetic tests
def selftest() -> dict:
    out = {}
    ref = np.array([0] * 20 + [3] * 8 + [0] * 20)                                     # a short true A5C segment
    base = np.zeros(48, int); alt = ref.copy(); zs = zones(base, alt)
    u = zone_utilities(base, alt, ref, zs); out["two_cuts_recover_short"] = zs == [(20, 28)] and u[0]["positive"] and u[0]["dtp"] == 3 and u[0]["f1_after"] == 1.0
    ref2 = np.zeros(48, int); base2 = ref.copy(); alt2 = np.zeros(48, int); zs2 = zones(base2, alt2); u2 = zone_utilities(base2, alt2, ref2, zs2)
    out["remove_false_short"] = zs2 == [(20, 28)] and u2[0]["positive"] and u2[0]["dfp"] == -3 and u2[0]["f1_after"] == 1.0
    ref3 = np.array([0] * 24 + [1] * 24); base3 = np.array([0] * 26 + [1] * 22); alt3 = np.array([0] * 23 + [1] * 25); zs3 = zones(base3, alt3)
    out["boundary_shift_zone"] = zs3 == [(23, 26)] and np.array_equal(apply(base3, alt3, zs3, [True]), alt3)
    base4 = np.array([1, 1, 0, 0, 0, 0]); alt4 = np.array([2, 2, 0, 0, 0, 3]); zs4 = zones(base4, alt4, merge=1)
    out["sequence_ends"] = zs4 == [(0, 2), (5, 6)] and np.array_equal(apply(base4, alt4, zs4, [True, True]), alt4) and np.array_equal(apply(base4, alt4, zs4, [False, False]), base4)
    out["no_disagreement"] = zones(base, base) == [] and np.array_equal(apply(base, base, [], []), base)
    b5 = np.zeros(60, int); a5 = b5.copy(); a5[10:13] = 1; a5[25:28] = 1; a5[50:52] = 2
    out["merge_neighbours"] = zones(b5, a5) == [(10, 28), (50, 52)] and np.array_equal(apply(b5, a5, zones(b5, a5), [True, False])[10:28], a5[10:28])
    out["ok"] = all(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        r = selftest(); print(r)
        if not r["ok"]:
            raise SystemExit("a synthetic zone test failed")


if __name__ == "__main__":
    main()
