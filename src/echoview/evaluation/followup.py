"""Three checks on the development benchmark that follow the comparison (docs/FOLLOWUP.md).

  python -m echoview.evaluation.followup logits --images <EV9V Images> --out dense_logits.npz
  python -m echoview.evaluation.followup study --images <EV9V Images> --logits <development logits> \\
      --dense dense_logits.npz --store EchoViewCLIP=<folder> --out followup.json

Q1 asks the window model more often than every 0.5 s. Q2 adds one number to the A5C logit. Q3 mixes EchoViewCLIP's
probabilities in. Each is chosen on the calibration half and reported on the selection half against the published
system, with the numbers and the paired resamples of the comparison (comparison.py). The plan fixes when a change is
called a candidate. The test benchmark is never read here, and nothing here changes the published system.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ..model import CLASSES
from ..pipeline import Ensemble, probabilities_from_logits
from ..reference import B3
from ..windows import nearest_centre, window_centres
from . import benchmark, comparison, ev9v, metrics, run
from .benchmark import Sequence
from .comparison import DESCRIBED, DRAWS, NATIVE_STATS, ROUTER, SEED, STREAM_STATS, Row, Scored
from .run import member_probabilities

PUBLISHED_ARM, INTERPOLATED = "every 0.5 s", "every 0.5 s, interpolated"
DENSER = {"every 0.2 s": 2, "every 0.1 s": 1}  # arm -> samples between two outputs
OFFSETS = tuple(float(b) for b in np.arange(-6, 7) / 2)  # -3.0 to +3.0 in steps of 0.5
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CLOSE = 1e-3  # the dense logits at the published centres against the stored logits
F1_MARGIN = 0.005
CANDIDATE, NOT_SUPPORTED, PUBLISHED_WORD = "candidate", "not supported on development data", "the published system"
A5C = CLASSES.index("A5C")
CURVE_NUMBERS = {"recordings": ("accuracy", "macro_f1", "a4c_as_a5c", "a5c_as_a4c", "recall_A4C", "recall_A5C"),
                 "multi": ("segment_f1_50", "piece_midpoint_A4C", "piece_midpoint_A5C", "wrong_samples", "cut_distance_mean_s")}

Probabilities = dict[str, np.ndarray]


# ------------------------------------------------------------------------------------------------- Q1: the read-out
def dense_check(stored: Mapping[str, np.ndarray], dense: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict:
    """The dense logits must hold one finite output for every sample and, at the published centres, the stored logits."""
    worst = 0.0
    for s in sequences:
        if dense[s.id].shape[0] != len(s):
            raise SystemExit("the dense logits do not hold one output for every sample of every sequence")
        if not (np.isfinite(dense[s.id]).all() and np.isfinite(stored[s.id]).all()):
            raise SystemExit("a logit of the dense or of the stored outputs is not finite")
        worst = max(worst, float(np.abs(dense[s.id][window_centres(len(s))] - stored[s.id]).max()))
    if worst > CLOSE:
        raise SystemExit(f"the dense logits do not agree with the stored ones at the published centres: {worst:.5f} apart")
    return {"sequences": len(sequences), "largest_difference": worst}


def interpolated(logits: np.ndarray, n_samples: int) -> np.ndarray:
    """The published outputs (windows, models, 5) -> per-sample probabilities (samples, 5), each model's probabilities
    interpolated linearly between two window centres. Before the first and after the last centre they stay flat."""
    centres, samples = window_centres(n_samples), np.arange(n_samples)
    by_window = torch.softmax(torch.from_numpy(np.asarray(logits, np.float32)), -1).numpy().astype(np.float64)  # as the published arm
    by_model = [np.stack([np.interp(samples, centres, by_window[:, m, v]) for v in range(by_window.shape[2])], 1) for m in range(by_window.shape[1])]
    return np.mean(by_model, 0)


def density_arms(stored: Mapping[str, np.ndarray], dense: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict[str, Probabilities]:
    """Per-sample probabilities of every arm of Q1. The published arm is always computed from the stored logits."""
    members = tuple(range(next(iter(stored.values())).shape[1]))
    arms = {PUBLISHED_ARM: member_probabilities(dict(stored), sequences, members),
            INTERPOLATED: {s.id: interpolated(stored[s.id], len(s)) for s in sequences}}
    for name, stride in DENSER.items():
        arms[name] = {s.id: probabilities_from_logits(dense[s.id][window_centres(len(s), stride)], len(s), stride) for s in sequences}
    return arms


def grid_labels(truth: np.ndarray, stride: int) -> np.ndarray:
    """A perfect system that is read every `stride` samples: the reference view at each centre, given to the samples
    nearest to it (the earlier centre on a tie, as everywhere in the package)."""
    centres = window_centres(len(truth), stride)
    return np.asarray(truth)[centres][nearest_centre(len(truth), centres)]


def forward_passes(sequences: tuple[Sequence, ...]) -> dict[str, float]:
    """How many windows an arm reads, as a multiple of the published system's."""
    published = sum(len(window_centres(len(s))) for s in sequences)
    return {PUBLISHED_ARM: 1.0, INTERPOLATED: 1.0,
            **{name: sum(len(window_centres(len(s), stride)) for s in sequences) / published for name, stride in DENSER.items()}}


# --------------------------------------------------------------------------------------------- Q2: the A5C offset
def with_offset(logits: Mapping[str, np.ndarray], offset: float) -> dict[str, np.ndarray]:
    """The logits with `offset` added to the A5C logit of every output of every model."""
    shift = np.zeros(len(CLASSES), np.float32)
    shift[A5C] = offset
    return {key: value + shift for key, value in logits.items()}


def pick_offset(curve: list[dict]) -> float:
    """The offset with the highest macro-F1; among equals the nearest to zero, and of two equally near the negative."""
    best = max(entry["native_macro_f1"] for entry in curve)
    return min((entry["offset"] for entry in curve if entry["native_macro_f1"] == best), key=lambda b: (abs(b), b))


def choose_offset(logits: Mapping[str, np.ndarray], calibration: tuple[Sequence, ...]) -> dict:
    """Q2 on the calibration half: the macro-F1 of its single recordings for every offset, decoder as released."""
    members = tuple(range(next(iter(logits.values())).shape[1]))
    curve = []
    for offset in OFFSETS:
        labels = comparison.decoded(member_probabilities(with_offset(logits, offset), calibration, members), B3)
        curve.append({"offset": offset, "native_macro_f1": float(metrics.evaluate(labels, calibration)["native_macro_f1"])})
    return {"chosen": pick_offset(curve), "curve": curve}


# -------------------------------------------------------------------------------------- Q3: another system mixed in
def mixed(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], weight: float) -> Probabilities:
    """(1 - weight) of the first system's probabilities and `weight` of the second's, whose rows are rescaled to 1."""
    out = {}
    for key, p in first.items():
        total = second[key].sum(1, keepdims=True)
        if (total <= 0).any():
            raise SystemExit("a stored row of the other system is without any probability for the five views")
        out[key] = (1.0 - weight) * np.asarray(p, float) + weight * (second[key] / total)
    return out


def choose_weight(router: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], calibration: tuple[Sequence, ...]) -> dict:
    """Q3 on the calibration half: every weight gets its decoder setting by the rule; the weight with the highest
    segment F1 at 50% on multi streams wins among those with a setting that meets the floors, the smaller of equals."""
    curve = [{"weight": w, **comparison.choose(mixed(router, other, w), calibration, drop_floors=False)} for w in WEIGHTS]
    feasible = [entry for entry in curve if entry["status"] == "feasible"]
    if not feasible:
        return {"chosen": 0.0, "curve": curve}
    best = max(entry["calibration"]["segment_f1_50_multi"] for entry in feasible)
    return {"chosen": min(e["weight"] for e in feasible if e["calibration"]["segment_f1_50_multi"] == best), "curve": curve}


# ------------------------------------------------------------------------------------------------------ the words
def _below_zero(entry: Mapping) -> bool:
    return entry["interval"] is not None and entry["interval"][1] < 0


def _above_zero(entry: Mapping) -> bool:
    return entry["interval"] is not None and entry["interval"][0] > 0


def word_q1(difference: Mapping | None) -> str:
    """An arm minus the published one on the selection half: nearer cuts, segment F1 not lower by more than the
    margin, macro-F1 of single recordings not lower. None: the rule found no setting for the arm."""
    if difference is None:
        return NOT_SUPPORTED
    nearer = _below_zero(difference["multi"]["cut_distance_mean_s"])
    kept = difference["multi"]["segment_f1_50"]["value"] >= -F1_MARGIN and difference["recordings"]["macro_f1"]["value"] >= 0
    return CANDIDATE if nearer and kept else NOT_SUPPORTED


def word_q2(chosen: float, difference: Mapping | None) -> str:
    if difference is None or chosen == 0:
        return NOT_SUPPORTED
    better = _above_zero(difference["recordings"]["macro_f1"]) and difference["multi"]["segment_f1_50"]["value"] >= -F1_MARGIN
    return CANDIDATE if better else NOT_SUPPORTED


def word_q3(chosen: float, difference: Mapping | None) -> str:
    if difference is None or chosen <= 0:
        return NOT_SUPPORTED
    better = _above_zero(difference["multi"]["segment_f1_50"]) and difference["recordings"]["macro_f1"]["value"] >= 0
    return CANDIDATE if better else NOT_SUPPORTED


# ------------------------------------------------------------------------------------------------------ the study
def _numbers(rows: list[Row], sequences: tuple[Sequence, ...]) -> dict[str, dict[str, float]]:
    """A few plain numbers of every row, without intervals: for the curves of Q2 and Q3."""
    scored = comparison.score_rows(rows, sequences, 1, SEED)
    return {row.name: {name: scored[row.name][part].values[name] for part, names in CURVE_NUMBERS.items() for name in names} for row in rows}


def _differences(scored: Mapping[str, Mapping[str, Scored]], name: str) -> dict:
    stats_of = {"multi": STREAM_STATS, "pairs": STREAM_STATS, "recordings": NATIVE_STATS}
    return {part: {stat.name: comparison.difference(scored[name][part], scored[ROUTER][part], stat.name, DESCRIBED) for stat in stats}
            for part, stats in stats_of.items()}


def _setting(chosen: Mapping) -> dict:
    return {key: chosen[key] for key in ("status", "n_feasible", "params", "calibration") if key in chosen}


def study(stored: Mapping[str, np.ndarray], dense: Mapping[str, np.ndarray], other: Mapping[str, Mapping[str, np.ndarray]] | None,
          halves: Mapping[str, tuple[Sequence, ...]], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The three questions. `stored` and `dense` hold the logits of both halves; `other` holds EchoViewCLIP's
    per-sample probabilities for both halves, or is None and Q3 is not run."""
    calibration, selection = halves["calibration"], halves["selection"]
    members = tuple(range(next(iter(stored.values())).shape[1]))
    published = {half: member_probabilities(dict(stored), sequences, members) for half, sequences in halves.items()}
    rows = [Row(ROUTER, ROUTER, "A", comparison.decoded(published["selection"], B3)),
            *(Row(f"Perfect on the {stride / metrics.HZ:.1f} s grid", "", "P", {s.id: grid_labels(s.labels, stride) for s in selection})
              for stride in (5, 2))]
    result: dict = {"benchmark": "development", "draws": draws, "seed": seed,
                    "dense_check": dense_check(stored, dense, (*calibration, *selection))}

    # Q1
    arms = {half: density_arms(stored, dense, sequences) for half, sequences in halves.items()}
    passes = forward_passes((*calibration, *selection))
    q1: dict = {"arms": {}}
    for arm in arms["calibration"]:
        chosen = comparison.choose(arms["calibration"][arm], calibration, drop_floors=False)
        q1["arms"][arm] = {"forward_passes": passes[arm], "setting": _setting(chosen)}
        if arm == PUBLISHED_ARM:
            continue
        rows.append(Row(f"{arm}, released numbers", ROUTER, "C", comparison.decoded(arms["selection"][arm], B3)))
        if chosen["status"] == "feasible":
            rows.append(Row(f"{arm}, own setting", ROUTER, "D", comparison.decoded(arms["selection"][arm], chosen["params"])))

    # Q2
    offset = choose_offset(stored, calibration)
    offset_rows = {half: [Row(f"{b:+.1f}", ROUTER, "D", comparison.decoded(member_probabilities(with_offset(stored, b), sequences, members), B3))
                          for b in OFFSETS] for half, sequences in halves.items()}
    curves = {half: _numbers(offset_rows[half], sequences) for half, sequences in halves.items()}
    q2: dict = {"chosen": offset["chosen"], "calibration_macro_f1": offset["curve"],
                "curve": {half: [{"offset": b, **curves[half][f"{b:+.1f}"]} for b in OFFSETS] for half in halves}}
    offset_name = f"A5C offset {offset['chosen']:+.1f}"
    if offset["chosen"] != 0:
        rows.append(Row(offset_name, ROUTER, "D", next(r.labels for r in offset_rows["selection"] if r.name == f"{offset['chosen']:+.1f}")))

    # Q3
    q3: dict = {"word": "not run: no stored outputs of EchoViewCLIP were given"}
    mix_name = ""
    if other is not None:
        weight = choose_weight(published["calibration"], other, calibration)
        mix_rows = [Row(f"{e['weight']:.1f}", ROUTER, "D", comparison.decoded(mixed(published["selection"], other, e["weight"]), e["params"]))
                    for e in weight["curve"]]
        on_selection = _numbers(mix_rows, selection)
        q3 = {"chosen": weight["chosen"],
              "curve": [{"weight": e["weight"], "setting": _setting(e), "selection": on_selection[f"{e['weight']:.1f}"]} for e in weight["curve"]]}
        mix_name = f"mixed with EchoViewCLIP, weight {weight['chosen']:.1f}"
        if weight["chosen"] > 0:
            rows.append(Row(mix_name, ROUTER, "D", next(r.labels for r in mix_rows if r.name == f"{weight['chosen']:.1f}")))

    scored = comparison.score_rows(rows, selection, draws, seed)
    result["rows"] = {row.name: {part: comparison.report(s) for part, s in scored[row.name].items()} for row in rows}
    result["differences_from_published"] = {row.name: _differences(scored, row.name) for row in rows if row.name != ROUTER}
    differences = result["differences_from_published"]
    for arm, entry in q1["arms"].items():
        entry["word"] = PUBLISHED_WORD if arm == PUBLISHED_ARM else word_q1(differences.get(f"{arm}, own setting"))
    q2["word"] = word_q2(offset["chosen"], differences.get(offset_name))
    if other is not None:
        q3["word"] = word_q3(q3["chosen"], differences.get(mix_name))
    result.update(q1=q1, q2=q2, q3=q3)
    return result


# ----------------------------------------------------------------------------------------------------- commands
def reference_system(weights: Path | None, device: str, batch: int) -> Ensemble:
    system = Ensemble(weights, device=device, batch=batch)
    if not system.is_reference:
        raise SystemExit("the loaded system is not the published system")
    return system


def _scored(halves: Mapping[str, tuple[Sequence, ...]]) -> dict[str, tuple[Sequence, ...]]:
    return {half: comparison._scored_kinds(halves[half]) for half in benchmark.HALVES}


def logits_command(args: argparse.Namespace) -> None:
    """The published system's logits for every sample of the development benchmark: one window per sample."""
    halves = _scored(benchmark.load(args.benchmark))
    sequences = tuple(s for half in benchmark.HALVES for s in halves[half])
    system = reference_system(args.weights, args.device, args.batch)
    table = ev9v.FrameTable.build(sequences, args.images, workers=args.workers)
    logits = {s.id: system.window_logits(table.frames(s), stride=1) for s in sequences}
    if args.out.exists():  # made while this ran
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    run.save_logits(logits, args.out)


def study_command(args: argparse.Namespace) -> None:
    halves = _scored(benchmark.load())
    sequences = tuple(s for half in benchmark.HALVES for s in halves[half])
    stored, dense = comparison._logits(args.logits, sequences), comparison._logits(args.dense, sequences)
    digests = {"logits": comparison.file_sha256(args.logits), "dense": comparison.file_sha256(args.dense)}
    f1 = metrics.evaluate(comparison.decoded(member_probabilities(stored, halves["selection"], tuple(range(next(iter(stored.values())).shape[1]))), B3),
                          halves["selection"])["segment_f1_50_multi"]
    if round(f1, 4) != comparison.PUBLISHED_SELECTION_F1:
        raise SystemExit(f"the stored logits must give {comparison.PUBLISHED_SELECTION_F1} on the selection half; got {f1:.4f}")
    other = None
    folders = comparison._stores(args.store)
    if set(folders) - {"EchoViewCLIP"}:
        raise SystemExit("the plan mixes EchoViewCLIP in and no other system")
    if folders:
        other, digests["EchoViewCLIP"] = comparison.read_store(folders["EchoViewCLIP"], sequences, comparison._names(args.images, sequences))
    result = study(stored, dense, other, halves, args.draws, SEED)
    comparison._write({"sha256": digests, "published_selection_f1": float(f1), **result}, args.out)


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.followup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("logits", help="one output of the published system for every sample of the development benchmark")
    p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V")
    p.add_argument("--benchmark", type=Path, default=None, help="a benchmark file; the packaged development benchmark when omitted")
    p.add_argument("--weights", type=Path, default=None, help="the folder of the published weights; downloaded when omitted")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="threads that decode frames")
    p.add_argument("--out", type=Path, required=True, help=".npz file to write; an existing file is never replaced")
    p.set_defaults(run=logits_command)
    p = commands.add_parser("study", help="the three questions, chosen on the calibration half and reported on the selection half")
    p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V; it gives the stored files of single recordings their names")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    p.add_argument("--dense", type=Path, required=True, help="the output of the logits command")
    p.add_argument("--store", action="append", default=[], help="EchoViewCLIP=FOLDER, its stored outputs for the development benchmark (Q3)")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=study_command)
    args = top.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    args.run(args)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
