"""What the parts of the published system add, on the development benchmark (docs/ABLATIONS.md, part 1).

  python -m echoview.evaluation.ablations --logits <development logits> --out stored.json

No model runs: the input is the window logits of the published system for both development halves, as
`echoview evaluate --save-logits` writes them. Four questions:

  the size of the ensemble   every subset of the three seeds, each with the decoder setting the rule gives it
  other decoders             no decoder, a mode filter, a mean filter, a minimum duration and a Viterbi path, each with
                             its one number chosen by the project's rule, against the released change-point decoder
  the decoder's settings     the released decoder at every setting of its grid
  one temperature            a single number that divides the logits, fitted for `classify`'s probabilities

Everything that is chosen is chosen on the calibration half and reported on the selection half. The test benchmark is
not read, and nothing here changes the published system.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from .. import decoder
from ..model import CLASSES
from ..pipeline import classification, softmax
from ..reference import B3
from . import benchmark, calibrate, classify_read, comparison, metrics
from .benchmark import Sequence, of_kind
from .comparison import DESCRIBED, DRAWS, ROUTER, SEED, Row, Scored
from .run import member_probabilities

N = len(CLASSES)
EPS = 1e-12
NONE, MODE, MEAN, DURATION, VITERBI = "no decoder", "mode filter", "mean filter", "minimum duration", "Viterbi path"
NO_VIEW_TEST = "change points without the view test"
# The one number of every other decoder, in the order in which equal scores are broken (the earlier wins).
GRIDS: dict[str, tuple[float, ...]] = {
    MODE: (3, 5, 9, 13, 17, 21, 25, 31),            # samples in the window whose most frequent view a sample takes
    MEAN: (3, 5, 9, 13, 17, 21, 25, 31),            # samples over which the probabilities are averaged
    DURATION: (3, 5, 8, 10, 15, 20, 30),            # samples a piece must last
    VITERBI: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),  # what a change of view costs, in units of log probability
}
DECODERS_LEVEL = 99.0  # five other decoders are compared with the released one, so each interval that decides is taken at 99%
COMPARED = (NONE, MODE, MEAN, DURATION, VITERBI)  # the five; the released decoder without one of its parts is described, not compared
AHEAD, BEHIND, NOT_SHOWN = "ahead of", "behind", "not shown to differ from"
TEMPERATURES = tuple(round(0.30 + 0.05 * k, 2) for k in range(35))  # 0.30 to 2.00
CANDIDATE, NOT_SUPPORTED = "candidate", "not supported on development data"
KEPT = {"multi": ("segment_f1_50", "edit", "wrong_samples", "cut_precision", "cut_recall", "cut_distance_mean_s", "same_view_cut_rate",
                  "extra_segments", "missed_segments"),
        "pairs": ("segment_f1_50",),
        "recordings": ("accuracy", "macro_f1", "a4c_as_a5c", "a5c_as_a4c", "cuts_per_recording")}


# ------------------------------------------------------------------------------------------------ other decoders
def _window_sums(values: np.ndarray, k: int) -> np.ndarray:
    """For every sample, the sum of `values` (samples, columns) over the k samples centred on it, cut off at the ends."""
    n, half = len(values), int(k) // 2
    cumulative = np.cumsum(np.pad(values, ((1, 0), (0, 0))), axis=0)
    at = np.arange(n)
    return cumulative[np.minimum(n, at + half + 1)] - cumulative[np.maximum(0, at - half)]


def mode_filter(prob: np.ndarray, k: float) -> np.ndarray:
    """Every sample takes the most frequent likeliest view among the k samples centred on it. Where its own view is
    among the most frequent it keeps it; otherwise the view with the lowest index of them."""
    labels = np.asarray(prob, float).argmax(1)
    counts = _window_sums(np.eye(N)[labels], k)
    own = counts[np.arange(len(labels)), labels] == counts.max(1)
    return np.where(own, labels, counts.argmax(1)).astype(int)


def mean_filter(prob: np.ndarray, k: float) -> np.ndarray:
    """The likeliest view of the probabilities summed over the k samples centred on a sample."""
    return _window_sums(np.asarray(prob, float), k).argmax(1).astype(int)


def minimum_duration(prob: np.ndarray, d: float) -> np.ndarray:
    """The likeliest view of every sample, and then no piece shorter than d samples: the shortest such piece (the
    earliest of equals) takes the view of its longer neighbour (the earlier of equals), until none is left."""
    labels = np.asarray(prob, float).argmax(1).astype(int)
    while True:
        runs = decoder.segments(labels)
        short = [(end - start, start, end) for start, end, _ in runs if end - start < d]
        if len(runs) <= 1 or not short:
            return labels
        _, start, end = min(short)
        neighbours = [(-(b - a), a, view) for a, b, view in runs if b == start or a == end]
        labels[start:end] = min(neighbours)[2]


def viterbi(prob: np.ndarray, penalty: float) -> np.ndarray:
    """The path of views with the largest sum of log probabilities, where every change of view costs `penalty`."""
    logp = np.log(np.maximum(np.asarray(prob, float), EPS))
    n = len(logp)
    score, back = logp[0].copy(), np.zeros((n, N), int)
    views = np.arange(N)
    for t in range(1, n):
        best = int(score.argmax())
        change = (score[best] - penalty) > score  # staying wins a tie
        back[t] = np.where(change, best, views)
        score = np.where(change, score[best] - penalty, score) + logp[t]
    path = np.empty(n, int)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


OTHERS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {MODE: mode_filter, MEAN: mean_filter, DURATION: minimum_duration, VITERBI: viterbi}


def decode_with(name: str, value: float, probabilities: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: OTHERS[name](p, value) for key, p in probabilities.items()}


def without_the_view_test(prob: np.ndarray, setting: Mapping) -> np.ndarray:
    """The released decoder without one of its parts: a peak of the score is a cut, whether or not the two sides
    name different views. Every piece takes the view with the largest mean probability, as in the decoder."""
    prob = np.asarray(prob, float)
    scores = decoder.boundary_scores(prob, int(setting["half_window"]))
    peaks = decoder.non_max_suppression(scores, float(setting["threshold"]), int(setting["nms_radius"]))
    cuts = np.concatenate([[0], peaks[peaks > 0], [len(prob)]]).astype(int)
    labels = np.empty(len(prob), dtype=int)
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end > start:
            labels[start:end] = int(prob[start:end].mean(0).argmax())
    return labels


def choose_by_the_rule(values: list, decode: Callable[[object, Mapping[str, np.ndarray]], dict[str, np.ndarray]],
                       probabilities: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict:
    """One of `values` by the project's rule on the calibration half: among those that keep the two floors, the
    highest segment F1@50 on the multi streams, the earlier of equals. When none keeps the floors they are dropped,
    and the output says so."""
    records = []
    for order, value in enumerate(values):
        result = metrics.evaluate(decode(value, probabilities), sequences)
        records.append({"order": order, "value": value, **{key: result[key] for key in calibrate.REPORTED}})
    chosen = calibrate.select_setting(records)
    feasible = chosen["chosen"] is not None
    best = chosen["chosen"] if feasible else max(records, key=lambda r: (r["segment_f1_50_multi"], -r["order"]))
    return {"value": best["value"], "status": chosen["status"], "floors_dropped": not feasible, "n_feasible": chosen["n_feasible"],
            "calibration": {key: best[key] for key in calibrate.REPORTED}, "grid": len(values)}


def choose_number(name: str, probabilities: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict:
    """The one number of another decoder, from its grid, by the project's rule."""
    return choose_by_the_rule(list(GRIDS[name]), lambda value, given: decode_with(name, value, given), probabilities, sequences)


# ------------------------------------------------------------------------------------------------------ numbers
def numbers(scored: Mapping[str, Scored]) -> dict:
    """The numbers this study keeps of a row, as the comparison reports them."""
    return {part: {key: entry for key, entry in comparison.report(scored[part]).items() if key in KEPT[part]} for part in KEPT}


def minus(first: Mapping[str, Scored], second: Mapping[str, Scored], level: float = DESCRIBED) -> dict:
    """First minus second for every kept number, with the same draws for both."""
    return {part: {key: comparison.difference(first[part], second[part], key, level) for key in KEPT[part]} for part in KEPT}


def word(difference: Mapping) -> str:
    """The released decoder against another one: the interval of segment F1@50 on the multi streams decides."""
    interval = difference["interval"]
    if interval is None:
        return NOT_SHOWN
    return AHEAD if interval[0] > 0 else BEHIND if interval[1] < 0 else NOT_SHOWN


def subset_name(members: tuple[int, ...]) -> str:
    return ("seed " if len(members) == 1 else "seeds ") + "+".join(str(m) for m in members)


def classify_numbers(stored: Mapping[str, np.ndarray], natives: tuple[Sequence, ...], members: tuple[int, ...]) -> dict:
    """What `classify`'s rule gives for the system of the given members: wrong recordings and the calibration error."""
    decided = {s.id: classification(stored[s.id][:, list(members)]) for s in natives}
    found = classify_read.described(decided, natives, draws=1, seed=SEED)
    return {"errors": found["errors"], "ece": found["probability"]["ece"], "mean_probability": found["probability"]["mean_probability"]}


# ------------------------------------------------------------------------------------------- the four questions
def ensemble_size(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int, seed: int) -> dict:
    """Every subset of the seeds as a system of its own, with the decoder setting the rule gives it on the
    calibration half, reported on the selection half and against the whole ensemble with its own setting."""
    calibration, selection = halves["calibration"], halves["selection"]
    count = next(iter(stored.values())).shape[1]
    subsets = [members for size in range(1, count + 1) for members in itertools.combinations(range(count), size)]
    rows, chosen = [], {}
    for members in subsets:
        name = subset_name(members)
        chosen[name] = comparison.choose(member_probabilities(dict(stored), calibration, members), calibration, drop_floors=False)
        rows.append(Row(name, ROUTER, "S", comparison.decoded(member_probabilities(dict(stored), selection, members), chosen[name]["params"])))
    scored = comparison.score_rows(rows, selection, draws, seed)
    whole = subset_name(tuple(range(count)))
    out = {}
    for members, row in zip(subsets, rows):
        params = chosen[row.name]["params"]
        out[row.name] = {"members": list(members), "setting": {"status": chosen[row.name]["status"], **params},
                         "is_the_published_setting": all(params[key] == B3[key] for key in ("half_window", "nms_radius"))
                         and abs(params["threshold"] - B3["threshold"]) < comparison.THRESHOLD_CLOSE,
                         "selection": numbers(scored[row.name]), "minus_the_whole_ensemble": minus(scored[row.name], scored[whole]),
                         "classify": classify_numbers(stored, of_kind(selection, "native"), members)}
    sizes = {size: [out[subset_name(m)]["selection"]["multi"]["segment_f1_50"]["value"] for m in subsets if len(m) == size] for size in range(1, count + 1)}
    return {"systems": out, "mean_segment_f1_50_multi_by_size": {str(size): float(np.mean(values)) for size, values in sizes.items()}}


def other_decoders(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int, seed: int) -> dict:
    """The released decoder against five others on the ensemble's probabilities."""
    calibration, selection = halves["calibration"], halves["selection"]
    members = tuple(range(next(iter(stored.values())).shape[1]))
    probabilities = {half: member_probabilities(dict(stored), sequences, members) for half, sequences in halves.items()}
    rows = [Row(ROUTER, ROUTER, "A", comparison.decoded(probabilities["selection"], B3)),
            Row(NONE, ROUTER, "N", comparison.likeliest(probabilities["selection"]))]
    chosen = {}
    for name in OTHERS:
        chosen[name] = choose_number(name, probabilities["calibration"], calibration)
        rows.append(Row(name, ROUTER, "D", decode_with(name, chosen[name]["value"], probabilities["selection"])))
    # the released decoder without its view test, over the decoder's own grid
    grid = calibrate.grid([probabilities["calibration"][s.id] for s in of_kind(calibration, "native")])
    chosen[NO_VIEW_TEST] = choose_by_the_rule(grid, lambda setting, given: {key: without_the_view_test(p, setting) for key, p in given.items()},
                                              probabilities["calibration"], calibration)
    rows.append(Row(NO_VIEW_TEST, ROUTER, "D", {key: without_the_view_test(p, chosen[NO_VIEW_TEST]["value"]) for key, p in probabilities["selection"].items()}))
    scored = comparison.score_rows(rows, selection, draws, seed)
    out = {"released": {"setting": dict(B3), "selection": numbers(scored[ROUTER])}, "others": {}}
    for row in rows[1:]:
        entry = {"chosen": chosen.get(row.name), "selection": numbers(scored[row.name]), "released_minus_this": minus(scored[ROUTER], scored[row.name])}
        if row.name in COMPARED:
            entry["decides"] = minus(scored[ROUTER], scored[row.name], DECODERS_LEVEL)["multi"]["segment_f1_50"]
            entry["word"] = word(entry["decides"])
        out["others"][row.name] = entry
    return out


def decoder_settings(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]]) -> dict:
    """The released decoder at every setting of its grid: the grid of the rule, with thresholds from the calibration
    half, and the three numbers of the rule on both halves."""
    members = tuple(range(next(iter(stored.values())).shape[1]))
    probabilities = {half: member_probabilities(dict(stored), sequences, members) for half, sequences in halves.items()}
    grid = calibrate.grid([probabilities["calibration"][s.id] for s in of_kind(halves["calibration"], "native")])
    table = []
    for order, setting in enumerate(grid):
        entry = {"order": order, **setting}
        for half, sequences in halves.items():
            result = metrics.evaluate(calibrate.decode_all(setting, probabilities[half]), sequences)
            entry[half] = {key: result[key] for key in calibrate.REPORTED}
        entry["feasible_on_calibration"] = bool(entry["calibration"]["native_macro_f1"] >= calibrate.NATIVE_FLOOR
                                                and (entry["calibration"]["boundary_recall_multi"] or 0.0) >= calibrate.RECALL_FLOOR)
        table.append(entry)
    published = metrics.evaluate(calibrate.decode_all(B3, probabilities["selection"]), halves["selection"])["segment_f1_50_multi"]
    scores = np.array([entry["selection"]["segment_f1_50_multi"] for entry in table])
    feasible = np.array([entry["feasible_on_calibration"] for entry in table])
    # the two halves with their parts exchanged: the rule chooses on the selection half, and the calibration half reports
    exchanged = comparison.choose(probabilities["selection"], halves["selection"], drop_floors=False)
    on_calibration = {name: metrics.evaluate(calibrate.decode_all(setting, probabilities["calibration"]), halves["calibration"])
                      for name, setting in (("chosen_on_selection", exchanged["params"]), ("published", B3))}
    return {"settings": table, "published_selection": float(published),
            "halves_exchanged": {"status": exchanged["status"], "setting": exchanged["params"],
                                 "calibration": {name: {key: result[key] for key in calibrate.REPORTED} for name, result in on_calibration.items()}},
            "selection_segment_f1_50_multi": {"lowest": float(scores.min()), "median": float(np.median(scores)), "highest": float(scores.max()),
                                              "within_0.01_of_the_published": int((np.abs(scores - published) <= 0.01).sum()),
                                              "within_0.02_of_the_published": int((np.abs(scores - published) <= 0.02).sum()),
                                              "above_the_published": int((scores > published).sum()), "settings": len(scores)},
            "feasible_on_calibration": {"settings": int(feasible.sum()), "lowest_on_selection": float(scores[feasible].min()) if feasible.any() else None,
                                        "highest_on_selection": float(scores[feasible].max()) if feasible.any() else None}}


def recording_probabilities(stored: Mapping[str, np.ndarray], natives: tuple[Sequence, ...], temperature: float) -> np.ndarray:
    """(recordings, 5): `classify`'s five probabilities when every logit is divided by the temperature."""
    return np.stack([softmax(np.asarray(stored[s.id], np.float64) / temperature).mean(0).mean(0) for s in natives])


def log_losses(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(probabilities[np.arange(len(truth)), truth], EPS))


def one_temperature(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int, seed: int) -> dict:
    """One temperature for all logits, fitted on the single recordings of the calibration half (the lowest log loss
    of `classify`'s probabilities; of equals the one nearest to 1) and reported on those of the selection half."""
    natives = {half: of_kind(sequences, "native") for half, sequences in halves.items()}
    truth = {half: np.array([int(s.labels[0]) for s in natives[half]]) for half in natives}
    curve = [{"temperature": t, "log_loss": float(log_losses(recording_probabilities(stored, natives["calibration"], t), truth["calibration"]).mean())}
             for t in TEMPERATURES]
    chosen = min(curve, key=lambda entry: (entry["log_loss"], abs(entry["temperature"] - 1.0)))["temperature"]
    unit = classify_read.units(natives["selection"])
    counts = classify_read.drawn(unit, draws, seed)
    out: dict = {"chosen": chosen, "calibration_curve": curve, "selection": {}}
    per_recording = {}
    for name, t in (("as published", 1.0), ("with the temperature", chosen)):
        probabilities = recording_probabilities(stored, natives["selection"], t)
        part = classify_read.probability_part(probabilities, truth["selection"], unit, draws, seed)
        right = probabilities.argmax(1) == truth["selection"]
        out["selection"][name] = {"temperature": t, "errors": int((~right).sum()), **{key: part[key] for key in (
            "ece", "ece_equal_count", "mean_probability", "share_right", "probability_minus_right", "brier", "log_loss", "median_probability")}}
        per_recording[name] = {"log_loss": log_losses(probabilities, truth["selection"]),
                               "brier": ((probabilities - np.eye(N)[truth["selection"]]) ** 2).sum(1)}
    out["with_minus_published"] = {key: classify_read.mean_interval(per_recording["with the temperature"][key] - per_recording["as published"][key], counts)
                                   for key in ("log_loss", "brier")}
    interval = out["with_minus_published"]["log_loss"]["interval"]
    more_wrong = out["selection"]["with the temperature"]["errors"] > out["selection"]["as published"]["errors"]
    out["word"] = CANDIDATE if chosen != 1.0 and interval is not None and interval[1] < 0 and not more_wrong else NOT_SUPPORTED
    return out


def study(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The four questions of part 1."""
    return {"benchmark": "development", "draws": draws, "seed": seed,
            "sequences": {half: len(sequences) for half, sequences in halves.items()},
            "ensemble_size": ensemble_size(stored, halves, draws, seed), "decoders": other_decoders(stored, halves, draws, seed),
            "decoder_settings": decoder_settings(stored, halves), "temperature": one_temperature(stored, halves, draws, seed)}


# ----------------------------------------------------------------------------------------------------- command
def held_to_published(stored: Mapping[str, np.ndarray], selection: tuple[Sequence, ...]) -> float:
    """The stored logits must give the published segment F1@50 of the selection half. Stops when they do not."""
    members = tuple(range(next(iter(stored.values())).shape[1]))
    f1 = metrics.evaluate(comparison.decoded(member_probabilities(dict(stored), selection, members), B3), selection)["segment_f1_50_multi"]
    if round(f1, 4) != comparison.PUBLISHED_SELECTION_F1:
        raise SystemExit(f"the stored logits must give {comparison.PUBLISHED_SELECTION_F1} on the selection half; got {f1:.4f}")
    return float(f1)


def development_halves() -> dict[str, tuple[Sequence, ...]]:
    return {half: comparison._scored_kinds(sequences) for half, sequences in benchmark.load().items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m echoview.evaluation.ablations", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    halves = development_halves()
    stored = comparison._logits(args.logits, tuple(s for half in benchmark.HALVES for s in halves[half]))
    f1 = held_to_published(stored, halves["selection"])
    result = {"sha256": {"logits": comparison.file_sha256(args.logits)}, "published_selection_f1": f1, **study(stored, halves, args.draws, SEED)}
    comparison._write(result, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
