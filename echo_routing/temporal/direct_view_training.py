"""A51-P deterministic exposure and durable, once-only row inference."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def source_plan(directory: np.ndarray, updates=600) -> np.ndarray:
    directory = np.asarray(directory)
    rng = np.random.default_rng(2026110151)
    pools = [np.flatnonzero(directory == j) for j in (0, 1)]
    if any(not len(p) or len(p) > 4 * updates for p in pools):
        raise ValueError("complete source coverage cannot fit fixed plan")
    streams = []
    for pool in pools:
        stream = []
        while len(stream) < updates * 4:
            stream.extend(rng.permutation(pool).tolist())
        streams.append(np.asarray(stream[:updates * 4]).reshape(updates, 4))
    plan = np.concatenate(streams, axis=1)
    assert set(plan.ravel()) == set(range(len(directory)))
    return plan


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_cell(root, key, identity, n):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    progress, data = root / (key + ".progress.json"), root / (key + ".npy")
    if progress.exists():
        rec = json.loads(progress.read_text())
        if rec["identity"] != identity or rec["n"] != n:
            raise ValueError("score-cell identity mismatch")
        if rec["pending"] is not None:
            raise RuntimeError("inflight possibly executed row cannot be repeated")
        if not data.exists():
            raise RuntimeError("claimed cell has no durable data array")
        scores = np.load(data, mmap_mode="r+")
        if scores.shape != (n,) or not np.isfinite(scores[:rec["done"]]).all():
            raise RuntimeError("saved cell prefix is incomplete")
    else:
        if data.exists():
            raise RuntimeError("unclaimed score data already exists")
        rec = {"identity": identity, "n": n, "done": 0, "pending": None}
        with progress.open("x") as f:
            json.dump(rec, f); f.flush(); os.fsync(f.fileno())
        scores = np.lib.format.open_memmap(data, mode="w+", dtype=np.float64, shape=(n,))
        scores[:] = np.nan; scores.flush()
        with data.open("r+b") as f:
            os.fsync(f.fileno())
    return {"progress": progress, "data": data, "scores": scores, "record": rec,
            "receipt": root / (key + ".complete.json")}


def claim_row(cell):
    rec = cell["record"]
    if rec["pending"] is not None:
        raise RuntimeError("inflight row already claimed")
    if rec["done"] == rec["n"]:
        return None
    rec["pending"] = rec["done"]
    write_json(cell["progress"], rec)
    return rec["pending"]


def finish_row(cell, index, score):
    rec = cell["record"]
    if index != rec["done"] or rec["pending"] != index or not np.isfinite(score):
        raise ValueError("score row does not match exclusive pending claim")
    cell["scores"][index] = float(score); cell["scores"].flush()
    with cell["data"].open("r+b") as f:
        os.fsync(f.fileno())
    rec["done"] += 1; rec["pending"] = None
    write_json(cell["progress"], rec)


def seal_cell(cell):
    rec = cell["record"]
    if rec["pending"] is not None or rec["done"] != rec["n"] or not np.isfinite(cell["scores"]).all():
        raise RuntimeError("cannot seal incomplete score cell")
    receipt = {"identity": rec["identity"], "rows": rec["n"], "sha256": _sha(cell["data"])}
    if cell["receipt"].exists():
        if json.loads(cell["receipt"].read_text()) != receipt:
            raise ValueError("sealed cell rebound")
    else:
        write_json(cell["receipt"], receipt)
    return receipt


def source_metrics(y, scores, discordant):
    y = np.asarray(y, bool); pred = np.asarray(scores) > 0
    discordant = np.asarray(discordant, bool)
    if not y.any() or y.all() or not discordant.any():
        raise ValueError("source metrics require both classes and discordant requests")
    tp, tn = int((pred & y).sum()), int((~pred & ~y).sum())
    fp, fn = int((pred & ~y).sum()), int((~pred & y).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
            "accuracy": float((pred == y).mean()), "a4c_recall": tn / (tn + fp),
            "a5c_recall": tp / (tp + fn), "a4c_fpr": fp / (tn + fp),
            "discordant_accuracy": float((pred[discordant] == y[discordant]).mean())}
