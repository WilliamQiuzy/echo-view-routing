"""Three checks on the development benchmark that follow the comparison, and the confirmation of one of them on the
test benchmark (docs/FOLLOWUP.md).

  python -m echoview.evaluation.followup logits --images <EV9V Images> --out dense_logits.npz
  python -m echoview.evaluation.followup study --images <EV9V Images> --logits <development logits> \\
      --dense dense_logits.npz --store EchoViewCLIP=<folder> --out followup.json
  python -m echoview.evaluation.followup logits --test --images <EV9V Images> --out dense_test_logits.npz
  python -m echoview.evaluation.followup confirm --logits <test logits> --dense dense_test_logits.npz \\
      --earlier docs/comparison/test.json --out confirmation.json

Q1 asks the window model more often than every 0.5 s. Q2 adds one number to the A5C logit. Q3 mixes EchoViewCLIP's
probabilities in. Each is chosen on the calibration half and reported on the selection half against the published
system, with the numbers and the paired resamples of the comparison (comparison.py). The plan fixes when a change is
called a candidate. `study` never reads the test benchmark. `logits` reads it only with `--test`: its images, and no
label.

`confirm` is the one read of the test benchmark that section 9 of the plan allows: the two denser read-outs of Q1, with
their released settings, against the published system. It first holds the published system's rows to the comparison's
first read, and once its output exists it can only reproduce it. Nothing here changes the published system.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ..model import CLASSES
from ..pipeline import Ensemble, probabilities_from_logits
from ..reference import B3, DENSER_READ_OUT
from ..windows import nearest_centre, window_centres
from . import benchmark, comparison, ev9v, metrics, run
from .benchmark import Sequence, of_kind
from .comparison import DESCRIBED, DRAWS, NATIVE_STATS, ROUTER, SEED, STREAM_STATS, Row, Scored
from .run import member_probabilities

PUBLISHED_ARM, INTERPOLATED = "every 0.5 s", "every 0.5 s, interpolated"
DENSER = {"every 0.2 s": 2, "every 0.1 s": 1}  # arm -> samples between two outputs
OFFSETS = tuple(float(b) for b in np.arange(-6, 7) / 2)  # -3.0 to +3.0 in steps of 0.5
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CLOSE = 1e-3  # the dense logits at the published centres against the stored logits
F1_MARGIN = 0.005
CANDIDATE, NOT_SUPPORTED, PUBLISHED_WORD = "candidate", "not supported on development data", "the published system"
# The confirmation of Q1 on the test benchmark (docs/FOLLOWUP.md, section 9).
CONFIRMED, AT_A_COST, NOT_CONFIRMED = "confirmed on the test benchmark", "nearer cuts at a cost", "not confirmed on the test benchmark"
ARM_LEVEL = 97.5  # two arms are tested, so each interval that decides is taken at 97.5%
HELD_ROWS = (ROUTER, "Perfect on the 0.5 s grid")  # rows of the comparison's first read that must come out again
# The outputs of the published models for every sample of the test benchmark (`logits --test`): the digest of that file
# is committed before the read (docs/followup/dense_test_check.json), and the read takes no other file.
TEST_DENSE_SHA256: str | None = "d1a23a82577b9af9fbade07c4f40aca64e081647c0e40d9af057e63f8bc3e500"
CONFIRMATION_SHA256: str | None = "87bd05802365859c63c9279cb6aa6d02b2a4ee1d32a71434d6e3c49538a0f5a2"  # docs/followup/confirmation.json, read on 2026-11-25
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
    return {PUBLISHED_ARM: member_probabilities(dict(stored), sequences, members),
            INTERPOLATED: {s.id: interpolated(stored[s.id], len(s)) for s in sequences}, **denser_arms(dense, sequences)}


def denser_arms(dense: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict[str, Probabilities]:
    """Per-sample probabilities of the arms that are read more often than every 0.5 s, from one output per sample."""
    return {name: {s.id: probabilities_from_logits(dense[s.id][window_centres(len(s), stride)], len(s), stride) for s in sequences}
            for name, stride in DENSER.items()}


def grid_labels(truth: np.ndarray, stride: int) -> np.ndarray:
    """A perfect system that is read every `stride` samples: the reference view at each centre, given to the samples
    nearest to it (the earlier centre on a tie, as everywhere in the package)."""
    centres = window_centres(len(truth), stride)
    return np.asarray(truth)[centres][nearest_centre(len(truth), centres)]


def perfect_row(sequences: tuple[Sequence, ...], stride: int) -> Row:
    return Row(f"Perfect on the {stride / metrics.HZ:.1f} s grid", "", "P", {s.id: grid_labels(s.labels, stride) for s in sequences})


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
    rows = [Row(ROUTER, ROUTER, "A", comparison.decoded(published["selection"], B3)), perfect_row(selection, 5), perfect_row(selection, 2)]
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


# ------------------------------------------------------------------------ the confirmation on the test benchmark
def word_confirm(decides: Mapping, f1: Mapping, macro: Mapping) -> dict:
    """The word for one arm, from the arm minus the published system: (a) the mean distance of the matched cuts is
    lower and the hull of its intervals by unit lies wholly below zero, (b) segment F1 at 50% on multi streams is not
    lower by more than the margin, (c) the macro-F1 of the single recordings is not lower."""
    nearer = decides["value"] is not None and decides["value"] < 0 and decides["hull"] is not None and decides["hull"][1] < 0
    f1_kept = f1["value"] is not None and f1["value"] >= -F1_MARGIN
    recordings_kept = macro["value"] is not None and macro["value"] >= 0
    word = NOT_CONFIRMED if not nearer else CONFIRMED if f1_kept and recordings_kept else AT_A_COST
    return {"nearer_cuts": nearer, "segment_f1_kept": f1_kept, "recordings_kept": recordings_kept, "word": word}


def published_numbers(labels: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict[str, float]:
    """The published numbers of the test benchmark from the published system's labels. Stops when one differs."""
    numbers = metrics.evaluate(labels, sequences)
    got = {key: float(numbers[key]) for key in comparison.PUBLISHED[ROUTER]}
    off = {key: got[key] for key, value in comparison.PUBLISHED[ROUTER].items() if abs(got[key] - value) > comparison.CLOSE}
    if off:
        raise SystemExit(f"the stored logits do not give the published numbers; nothing was scored. Differing: {off}")
    return got


def _plain(value: object) -> object:
    """The value as a JSON file gives it back."""
    return json.loads(json.dumps(comparison._clean(value), allow_nan=False))


def _reported(rows: list[Row], scored: Mapping[str, Mapping[str, Scored]]) -> dict[str, dict]:
    """Every row as the comparison's output holds it."""
    return {row.name: {"system": row.system, "role": row.role, **{part: comparison.report(s) for part, s in scored[row.name].items()}} for row in rows}


def held_to_first_read(first_read: object, rows: Mapping[str, Mapping], draws: int, seed: int) -> list[str]:
    """The published system's rows, computed again, against the comparison's first read of the test benchmark: number
    by number, intervals included, with the same draws and seed. Stops when anything differs."""
    if (not isinstance(first_read, Mapping) or any(key not in first_read for key in ("benchmark", "draws", "seed", "rows"))
            or any(name not in first_read["rows"] for name in HELD_ROWS)):
        raise SystemExit("the earlier file is not an output of the comparison with the published system's rows")
    problems = [key for key, value in (("benchmark", "test"), ("draws", draws), ("seed", seed)) if first_read[key] != value]
    problems += [f"row: {name}" for name in HELD_ROWS if _plain(first_read["rows"][name]) != _plain(rows[name])]
    if problems:
        raise SystemExit("the comparison's first read does not come out again: " + "; ".join(problems))
    return list(HELD_ROWS)


def confirm(stored: Mapping[str, np.ndarray], dense: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...], first_read: object,
            draws: int = DRAWS, seed: int = SEED) -> dict:
    """The read of section 9: the two released denser read-outs against the published system on the test benchmark.

    The published system's rows come first, from the stored logits. They must give the published numbers and the rows
    of the comparison's first read; no arm is decoded before that. Then every row gets every number of the comparison,
    and each arm its word."""
    members = tuple(range(next(iter(stored.values())).shape[1]))
    held = [Row(ROUTER, ROUTER, "A", comparison.decoded(member_probabilities(dict(stored), sequences, members), B3)), perfect_row(sequences, 5)]
    result: dict = {"draws": draws, "seed": seed, "published_numbers_reproduced": published_numbers(held[0].labels, sequences)}
    result["first_read_rows_reproduced"] = held_to_first_read(first_read, _reported(held, comparison.score_rows(held, sequences, draws, seed)), draws, seed)
    arms, passes = denser_arms(dense, sequences), forward_passes(sequences)
    rows = [*held, perfect_row(sequences, 2),
            *(Row(name, ROUTER, "D", comparison.decoded(arms[name], DENSER_READ_OUT[stride])) for name, stride in DENSER.items())]
    scored = comparison.score_rows(rows, sequences, draws, seed)
    result["rows"] = _reported(rows, scored)
    result["differences_from_published"] = {name: _differences(scored, name) for name in DENSER}
    natives, by_name = of_kind(sequences, "native"), {row.name: row for row in rows}
    for name in DENSER:  # the single recordings that only one of the two reads right
        only = comparison.discordant(natives, by_name[name], by_name[ROUTER])
        result["differences_from_published"][name]["recordings_only_one_reads_right"] = {
            "only_the_arm": only["only_first_right"], "only_the_published_system": only["only_second_right"]}
    result["confirmation"] = {}
    for name in DENSER:
        described = result["differences_from_published"][name]
        decides = comparison.difference(scored[name]["multi"], scored[ROUTER]["multi"], "cut_distance_mean_s", ARM_LEVEL)
        f1, macro = described["multi"]["segment_f1_50"], described["recordings"]["macro_f1"]
        result["confirmation"][name] = {"forward_passes": passes[name], "cut_distance": decides, "segment_f1_50": f1, "macro_f1": macro,
                                        **word_confirm(decides, f1, macro)}
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
    """The published system's logits for every sample of the development benchmark, or with --test of the test
    benchmark: one window per sample. Only images are read; nothing is scored."""
    if args.test:
        if args.benchmark is not None:
            raise SystemExit("--test reads the packaged test benchmark; give no --benchmark with it")
        sequences = comparison._scored_kinds(benchmark.load_test())
    else:
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


def confirm_command(args: argparse.Namespace) -> None:
    """The one read of the test benchmark for the denser read-out, with everything the plan fixes or not at all."""
    digests = {"logits": comparison.file_sha256(args.logits), "dense": comparison.file_sha256(args.dense),
               "first_read": comparison.file_sha256(args.earlier)}
    problems = []
    if args.draws != DRAWS:
        problems.append(f"the plan fixes {DRAWS} draws")
    if digests["logits"] != comparison.TEST_LOGITS_SHA256:
        problems.append("the logits file is not the one the plan names")
    if digests["first_read"] != comparison.FIRST_READ_SHA256:
        problems.append("--earlier must be the output of the comparison's first read, unchanged")
    if TEST_DENSE_SHA256 is None:
        problems.append("the code does not name the dense file yet: its digest is committed before the read")
    elif digests["dense"] != TEST_DENSE_SHA256:
        problems.append("the dense file is not the one whose digest was committed before the read")
    if CONFIRMATION_SHA256 is None and args.reproduce is not None:
        problems.append("there is no read of it to reproduce yet")
    if CONFIRMATION_SHA256 is not None and (args.reproduce is None or not args.reproduce.is_file()
                                            or comparison.file_sha256(args.reproduce) != CONFIRMATION_SHA256):
        problems.append("it has been read for this question, and a later run must be given that read's output, unchanged (--reproduce), "
                        "and can only reproduce it")
    if problems:
        raise SystemExit("the test benchmark is not read: " + "; ".join(problems))
    sequences = comparison._scored_kinds(benchmark.load_test())
    stored, dense = comparison._logits(args.logits, sequences), comparison._logits(args.dense, sequences)
    result = {"benchmark": "test", "sequences": len(sequences), "sha256": digests,
              "settings": {name: dict(DENSER_READ_OUT[stride]) for name, stride in DENSER.items()},
              "dense_check": dense_check(stored, dense, sequences),
              **confirm(stored, dense, sequences, json.loads(args.earlier.read_text()), args.draws, SEED)}
    if args.reproduce is not None and json.loads(args.reproduce.read_text()) != _plain(result):
        raise SystemExit("the read does not come out again; nothing is written")
    comparison._write(result, args.out)


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.followup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("logits", help="one output of the published system for every sample of the development benchmark, or of the test benchmark")
    p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V")
    p.add_argument("--benchmark", type=Path, default=None, help="a benchmark file; the packaged development benchmark when omitted")
    p.add_argument("--test", action="store_true", help="the packaged test benchmark instead: its images are read, no label, and nothing is scored")
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
    p = commands.add_parser("confirm", help="the one read of the test benchmark for the denser read-out (docs/FOLLOWUP.md, section 9)")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the test benchmark")
    p.add_argument("--dense", type=Path, required=True, help="the output of the logits command with --test")
    p.add_argument("--earlier", type=Path, required=True, help="the output of the comparison's first read of the test benchmark (docs/comparison/test.json)")
    p.add_argument("--reproduce", type=Path, help="the output of this read, once it exists: the result must equal it, or nothing is written")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=confirm_command)
    args = top.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    args.run(args)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
