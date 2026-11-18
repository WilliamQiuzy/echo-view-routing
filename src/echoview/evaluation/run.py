"""The development evaluation of the reference system: each seed alone and the three as one ensemble.

For every system the decoder setting is chosen on the calibration half by the project's rule and applied unchanged to
the selection half. The ensemble is also decoded with its published setting, which must give the same numbers. The
whole evaluation is one pass of the three seeds over the windows of both halves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..model import CLASSES
from ..pipeline import Ensemble, probabilities_from_logits, softmax
from ..reference import B3
from . import calibrate, metrics, scorecard
from .benchmark import HALVES, Sequence, of_kind
from .ev9v import FrameTable

Logits = dict[str, np.ndarray]  # sequence id -> (windows, models, 5)


def all_logits(system: Ensemble, sequences: tuple[Sequence, ...], table: FrameTable) -> Logits:
    return {sequence.id: system.window_logits(table.frames(sequence)) for sequence in sequences}


def member_probabilities(logits: Logits, sequences: tuple[Sequence, ...], members: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Per-sample probabilities of the system made of the given members (positions on the logits' model axis)."""
    return {s.id: probabilities_from_logits(logits[s.id][:, list(members)], len(s)) for s in sequences}


def expected_calibration_error(confidence: np.ndarray, right: np.ndarray, bins: int = 10) -> float:
    """Equal-width ECE of recording-level decisions."""
    confidence, right = np.asarray(confidence, float), np.asarray(right, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        member = (confidence > low) & (confidence <= high)
        if member.any():
            total += member.mean() * abs(right[member].mean() - confidence[member].mean())
    return float(total)


def recording_table(logits: Logits, natives: tuple[Sequence, ...], members: tuple[int, ...]) -> dict:
    """Whole recordings by the mean of their window probabilities: the recording-level routing decision."""
    k = len(CLASSES)
    confusion = np.zeros((k, k), np.int64)
    confidence, right = [], []
    for sequence in natives:
        mean = softmax(logits[sequence.id][:, list(members)]).mean((0, 1))
        truth = int(sequence.labels[0])
        confusion[truth, int(mean.argmax())] += 1
        confidence.append(float(mean.max()))
        right.append(int(mean.argmax()) == truth)
    recall = [float(confusion[c, c] / max(1, confusion[c].sum())) for c in range(k)]
    precision = [float(confusion[c, c] / max(1, confusion[:, c].sum())) for c in range(k)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    return {"recordings": int(confusion.sum()), "errors": int(confusion.sum() - np.trace(confusion)), "macro_f1": float(np.mean(f1)),
            "recall": recall, "confusion": confusion.tolist(), "median_confidence": float(np.median(confidence)),
            "ece": expected_calibration_error(np.asarray(confidence), np.asarray(right)),
            "a4c_to_a5c": int(confusion[2, 3]), "a5c_to_a4c": int(confusion[3, 2])}


def stream_scorecard(sequences: tuple[Sequence, ...], decoded: dict[str, np.ndarray], kind: str) -> dict | None:
    streams = [(s.labels, s.pieces, np.asarray(decoded[s.id], int)) for s in of_kind(sequences, kind)]
    return scorecard.summarise(streams) if streams else None


def half_report(sequences: tuple[Sequence, ...], logits: Logits, probabilities: dict[str, np.ndarray], setting: dict,
                members: tuple[int, ...]) -> dict:
    """Everything reported for one system on one half at one decoder setting."""
    decoded = calibrate.decode_all(setting, probabilities)
    scored = tuple(s for s in sequences if s.kind != "exam")
    return {"system": metrics.evaluate(decoded, scored),
            "native": recording_table(logits, of_kind(sequences, "native"), members),
            "streams_multi": stream_scorecard(sequences, decoded, "multi"),
            "streams_exam": stream_scorecard(sequences, decoded, "exam")}


def system_report(halves: dict[str, tuple[Sequence, ...]], logits: dict[str, Logits], members: tuple[int, ...],
                  fixed: dict | None = None) -> dict:
    """One system on both halves. Without `fixed` the decoder setting is chosen on the calibration half by the rule."""
    probabilities = {half: member_probabilities(logits[half], halves[half], members) for half in HALVES}
    if fixed is None:
        scored = {s.id: probabilities["calibration"][s.id] for s in halves["calibration"] if s.kind != "exam"}
        chosen = calibrate.choose(scored, tuple(s for s in halves["calibration"] if s.kind != "exam"))
        setting, how = chosen["params"], {k: v for k, v in chosen.items() if k != "best_unconstrained"}
    else:
        setting = {key: fixed[key] for key in ("half_window", "threshold", "nms_radius")}
        how = {"status": "fixed", "params": setting}
    report = {"b3": how}
    for half in HALVES:
        report[half] = half_report(halves[half], logits[half], probabilities[half], setting, members)
    return report


def evaluate_reference(system: Ensemble, halves: dict[str, tuple[Sequence, ...]], tables: dict[str, FrameTable],
                       logits: dict[str, Logits] | None = None) -> dict:
    """Each loaded seed alone, the ensemble with the rule's setting and the ensemble with its published setting.

    `logits` may hold window logits computed earlier (half -> sequence id -> array); without them the models are run.
    """
    if logits is None:
        logits = {half: all_logits(system, halves[half], tables[half]) for half in HALVES}
    names = [f"C2_s{seed}" for seed in system.seeds] or [f"model_{k}" for k in range(len(system.models))]
    everyone = tuple(range(len(system.models)))
    systems = {name: system_report(halves, logits, (k,)) for k, name in enumerate(names)}
    if len(everyone) > 1:
        systems["ensemble"] = system_report(halves, logits, everyone)
        systems["ensemble_published_setting"] = system_report(halves, logits, everyone, fixed=system.b3)
    return {"benchmark": "EV9V development halves (ev9v_dev.json.gz)", "model": system.version, "published_b3": dict(B3),
            "halves": {half: {"sequences": len(halves[half]), "frames": len(tables[half].refs)} for half in HALVES},
            "systems": systems}


HEADLINE = ("segment_f1_50_multi", "boundary_f1_multi", "native_macro_f1", "same_class_cut_rate_multi")


def headline(report: dict) -> list[str]:
    """One line per system: the numbers a reader looks at first."""
    lines = []
    for name, entry in report["systems"].items():
        sel, cal = entry["selection"]["system"], entry["calibration"]["system"]
        setting = entry["b3"]["params"]
        lines.append(f"{name:28s} selection F1@50 {sel['segment_f1_50_multi']:.4f}  calibration F1@50 {cal['segment_f1_50_multi']:.4f}  "
                     f"recording errors {entry['selection']['native']['errors']}/{entry['calibration']['native']['errors']}  "
                     f"decoder h{setting['half_window']} thr {setting['threshold']:.4f} ({entry['b3']['status']})")
    return lines


def write(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=1) + "\n")
