"""Writes metrics_parity.npz and metrics_parity.json: made-up predictions for every sequence of the EV9V development
benchmark, and what the research code's metrics, decoder rule and stream scorecard say about them.

Run inside the research repository (branch robust-upgrade, commit 157862d7 or later) on a machine that holds the
protocol files:  python make_metric_fixtures_from_research_code.py <research root> <output folder>
The predictions are random distortions of the reference labels, so no model output is stored. Sequences are keyed as in
the package's benchmark file: a native recording by its position (tools/research/export_ev9v_dev_benchmark.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R))
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from robust import devset, scorecard  # noqa: E402

rng = np.random.default_rng(20261118)


def distort(labels: np.ndarray) -> np.ndarray:
    """Reference labels with moved cuts, relabelled pieces and inserted short segments."""
    n = len(labels)
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [n]])
    out = np.empty(n, int)
    edge = 0
    for k, (a, b) in enumerate(zip(starts, ends)):
        stop = n if k == len(starts) - 1 else int(np.clip(b + rng.integers(-8, 9), edge + 1, n))
        out[edge:stop] = labels[a] if rng.random() > 0.08 else rng.integers(0, 5)
        edge = stop
    if edge < n:
        out[edge:] = labels[-1]
    if rng.random() < 0.2 and n > 6:
        a = int(rng.integers(0, n - 3))
        out[a:a + int(rng.integers(2, 20))] = rng.integers(0, 5)
    return out


def soft(labels: np.ndarray) -> np.ndarray:
    """Probabilities that mostly follow the distorted labels, with noise."""
    logits = rng.normal(0.0, 1.0, (len(labels), 5))
    logits[np.arange(len(labels)), labels] += rng.choice([1.5, 3.0, 5.0])
    prob = np.exp(logits - logits.max(-1, keepdims=True))
    return (prob / prob.sum(-1, keepdims=True)).astype(np.float32)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


arrays, report = {}, {}
for half in devset.HALVES:
    h = devset.Half(R, half)
    public = {s.seq_id: f"native_{half}_{k:04d}" for k, s in enumerate(h.ev.natives)}
    preds = {q["id"]: distort(np.asarray(q["labels"], int)) for q in h.seqs}
    exam_preds = {q["id"]: distort(np.asarray(q["labels"], int)) for q in h.exam}
    probs = {q["id"]: soft(preds[q["id"]]) for q in h.seqs}
    metrics, _ = evaluate_predictions(preds, h.ev, 10.0, 0.5)
    streams = {"multi": [], "exam": []}
    for q in h.seqs:
        if q["kind"] == "multi":
            video = np.array([r[0] for r in q["refs"]])
            streams["multi"].append((np.asarray(q["labels"], int), np.concatenate([[0], np.cumsum(video[1:] != video[:-1])]), preds[q["id"]]))
    for q in h.exam:
        video = np.array([r[0] for r in q["refs"]])
        streams["exam"].append((np.asarray(q["labels"], int), np.concatenate([[0], np.cumsum(video[1:] != video[:-1])]), exam_preds[q["id"]]))
    chosen = devset.choose_b3(probs, h)
    decoded_row = devset.system_row(h, probs, chosen["params"])
    report[half] = {"metrics": plain(metrics), "scorecard_multi": plain(scorecard.summarise(streams["multi"])),
                    "scorecard_exam": plain(scorecard.summarise(streams["exam"])),
                    "b3": plain({k: v for k, v in chosen.items()}), "decoded_metrics_at_chosen_setting": plain(decoded_row)}
    for sid, value in preds.items():
        arrays[f"{half}|pred|{public.get(sid, sid)}"] = value.astype(np.int8)
    for sid, value in exam_preds.items():
        arrays[f"{half}|exam|{sid}"] = value.astype(np.int8)
    for sid, value in probs.items():
        arrays[f"{half}|prob|{public.get(sid, sid)}"] = value
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "metrics_parity.npz", **arrays)
(OUT / "metrics_parity.json").write_text(json.dumps(report, indent=1) + "\n")
print({h: (report[h]["metrics"]["segment_f1_50_multi"], report[h]["b3"]["status"], report[h]["b3"]["params"]) for h in report})
