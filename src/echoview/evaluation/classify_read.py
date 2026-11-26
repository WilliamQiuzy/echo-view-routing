"""One read of the test benchmark for what `echoview classify` returns (docs/EVALUATION.md, "One read of the test
benchmark for `classify`").

  python -m echoview.evaluation.classify_read trial --logits <development logits> --report docs/evaluation/development_report.json --out trial.json
  python -m echoview.evaluation.classify_read read --logits <test logits> --earlier docs/comparison/test.json --trial trial.json --out classify_test.json

The published numbers of the test benchmark give a single recording the view of most of its decoded samples.
`echoview classify` takes the largest mean window probability instead, and returns that probability and whether the
seeds agree. `read` scores those three on the single recordings of the test benchmark, from the stored window logits
of the published system. No model runs.

`trial` never reads the test benchmark. It runs the same functions on the development halves, where the numbers of
this rule are published, and stops unless they come out. `read` takes the trial's output and does not start without it.

Before `read` scores anything, the stored logits must give the published numbers of the benchmark, and the published
system's rows must be those of the comparison's first read. Once the output of the read exists, `read` can only
reproduce it. Nothing here changes the published system, and the output holds pooled numbers only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..model import CLASSES
from ..pipeline import Classification, classification
from ..reference import B3
from . import benchmark, bootstrap, comparison, followup
from .benchmark import Sequence, of_kind
from .comparison import DESCRIBED, DRAWS, ROUTER, SEED, Row
from .followup_review import sign_test
from .run import expected_calibration_error, member_probabilities

MAJORITY, CLASSIFY = "majority of the decoded samples", "largest mean probability (classify)"
BINS = 10  # of equal width, as in the calibration error of the development report; and as many of equal count
COMPARED = ("accuracy", "macro_f1", "recall_A4C", "recall_A5C", "a4c_as_a5c", "a5c_as_a4c")
MORE, LESS, NOT_SHOWN = "more accurate than the majority rule", "less accurate than the majority rule", "not shown to differ from the majority rule"
P_BELOW = 0.05  # the word also needs the exact test of the recordings that only one rule reads right
TRIAL_SYSTEM = "ensemble_published_setting"  # the row of the development report that the trial is held to
CLOSE = 5e-5  # the trial: logits of another run of the same models differ in the sixth decimal, and so do numbers made of probabilities
# The output of the read (docs/evaluation/classify_test.json). None until the read is made; from then on `read` takes
# that file and can only reproduce it.
READ_SHA256: str | None = None
N = len(CLASSES)


# ------------------------------------------------------------------------------------------------ what is scored
def decisions(stored: Mapping[str, np.ndarray], natives: tuple[Sequence, ...]) -> dict[str, Classification]:
    """What `classify` returns for every single recording, from its window logits (windows, models, 5)."""
    return {s.id: classification(stored[s.id]) for s in natives}


def given_views(decided: Mapping[str, Classification], natives: tuple[Sequence, ...]) -> dict[str, int]:
    return {s.id: CLASSES.index(decided[s.id].view) for s in natives}


def majority_views(labels: Mapping[str, np.ndarray], natives: tuple[Sequence, ...]) -> dict[str, int]:
    """The view of most samples of every single recording: the rule of the benchmark's published numbers."""
    return {s.id: int(np.bincount(np.asarray(labels[s.id], int), minlength=N).argmax()) for s in natives}


def as_row(views: Mapping[str, int], sequences: tuple[Sequence, ...], published: Mapping[str, np.ndarray]) -> Row:
    """`classify`'s view as the view of every sample of its recording, so that the comparison's code scores it. The
    streams keep the published labels: that code scores every part, and only the recordings of this row are reported."""
    return Row(CLASSIFY, ROUTER, "R", {s.id: np.full(len(s), views[s.id], dtype=int) if s.kind == "native" else np.asarray(published[s.id], int)
                                       for s in sequences})


def confusion(views: Mapping[str, int], natives: tuple[Sequence, ...]) -> list[list[int]]:
    """Rows are the reference view, columns the view that is given."""
    table = np.zeros((N, N), np.int64)
    for s in natives:
        table[int(s.labels[0]), views[s.id]] += 1
    return table.tolist()


def units(natives: tuple[Sequence, ...]) -> np.ndarray:
    """The unit that is drawn for every recording: its examination, or the recording where the benchmark names none."""
    grouped = all(s.group >= 0 for s in natives)
    return np.array([s.group if grouped else k for k, s in enumerate(natives)], dtype=int)


# ---------------------------------------------------------------------------------------------- the probability
def _bin(probability: np.ndarray, right: np.ndarray, member: np.ndarray) -> dict:
    held = int(member.sum())
    return {"recordings": held, "right": int(right[member].sum()), "mean_probability": float(probability[member].mean()) if held else None,
            "share_right": float(right[member].mean()) if held else None}


def _gap(bins: list[dict]) -> float:
    """The bins' gaps between the share that is right and the mean probability, weighted by the bins' sizes."""
    total = sum(b["recordings"] for b in bins)
    return float(sum(b["recordings"] / total * abs(b["share_right"] - b["mean_probability"]) for b in bins if b["recordings"]))


def reliability(probability: np.ndarray, right: np.ndarray, bins: int = BINS) -> list[dict]:
    """The bins of the calibration error, of equal width: how many recordings each holds, how many of them are right,
    their mean probability and the share that is right. A bin without a recording has neither of the last two."""
    probability, right = np.asarray(probability, float), np.asarray(right, bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    return [{"above": float(low), "up_to": float(high), **_bin(probability, right, (probability > low) & (probability <= high))}
            for low, high in zip(edges[:-1], edges[1:])]


def equal_count(probability: np.ndarray, right: np.ndarray, bins: int = BINS) -> list[dict]:
    """The recordings in the order of their probability, cut into `bins` parts of as equal a size as there can be:
    each part with its lowest and highest probability and the numbers of `reliability`. Recordings of one probability
    stay in the order in which they were given, which is the benchmark's order."""
    probability, right = np.asarray(probability, float), np.asarray(right, bool)
    order = np.argsort(probability, kind="stable")
    out = []
    for part in np.array_split(order, bins):
        if len(part):
            member = np.zeros(len(probability), bool)
            member[part] = True
            out.append({"from": float(probability[part].min()), "to": float(probability[part].max()), **_bin(probability, right, member)})
    return out


def _pairs(probability: np.ndarray, right: np.ndarray) -> np.ndarray:
    """(right recordings, wrong recordings): 1 where the right one has the higher probability, 0.5 on a tie."""
    good, bad = probability[right][:, None], probability[~right][None, :]
    return (good > bad) + 0.5 * (good == bad)


def separation(probability: np.ndarray, right: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """The chance that a recording that is read right has a higher probability than one that is read wrong; a tie
    counts half (the area under the ROC curve of the probability as a sign of a right answer). `weights` says how
    often every recording counts. None when no recording is right or none is wrong."""
    probability, right = np.asarray(probability, float), np.asarray(right, bool)
    weights = np.ones(len(probability)) if weights is None else np.asarray(weights, float)
    good, bad = weights[right], weights[~right]
    if good.sum() <= 0 or bad.sum() <= 0:
        return None
    return float(good @ _pairs(probability, right) @ bad / (good.sum() * bad.sum()))


def drawn(unit: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """(draws, recordings): how often every recording counts in every resample of whole units. One generator started
    from the seed; the numbers about the probability share these draws."""
    unit = np.asarray(unit, int)
    return bootstrap.resampling_weights(np.random.default_rng(seed), int(unit.max()) + 1, draws)[:, unit]


def separation_interval(probability: np.ndarray, right: np.ndarray, counts: np.ndarray, level: float = DESCRIBED) -> dict:
    """The separation with a percentile interval over the resamples in `counts` (draws, recordings). A draw in which
    no recording is wrong, or none is right, gives no number; how many such draws there were is reported."""
    probability, right = np.asarray(probability, float), np.asarray(right, bool)
    value = separation(probability, right)
    if value is None:
        return {"value": None, "level": level, "interval": None, "draws_without_a_number": len(counts)}
    good, bad = counts[:, right], counts[:, ~right]
    top, bottom = ((good @ _pairs(probability, right)) * bad).sum(1), good.sum(1) * bad.sum(1)
    values = np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1.0), np.nan)
    return {"value": value, "level": level, "interval": comparison.percentiles(values, level), "draws_without_a_number": int(np.isnan(values).sum())}


def mean_interval(values: np.ndarray, counts: np.ndarray, level: float = DESCRIBED) -> dict:
    """The mean of one number per recording, with a percentile interval over the resamples in `counts`."""
    values = np.asarray(values, float)
    return {"value": float(values.mean()), "level": level, "interval": comparison.percentiles(counts @ values / counts.sum(1), level)}


def probability_part(probabilities: np.ndarray, truth: np.ndarray, unit: np.ndarray, draws: int, seed: int) -> dict:
    """What the probabilities that `classify` returns say. `probabilities` is (recordings, 5), `truth` the reference
    view of every recording. The probability of a recording is that of the view it is given, the largest of the five.

      ece, bins                  the calibration error of the development report: ten bins of equal width
      ece_equal_count, ...       the same gaps over ten parts of equal count
      probability_minus_right    the mean probability minus the share that is right; above zero: surer than right
      brier, log_loss            the mean squared distance of the five probabilities from the reference view (0 to 2),
                                 and the mean negative logarithm of the reference view's probability
      separation                 whether a wrong answer has a lower probability than a right one
    """
    probabilities, truth = np.asarray(probabilities, float), np.asarray(truth, int)
    probability, right = probabilities.max(1), probabilities.argmax(1) == truth
    counts = drawn(unit, draws, seed)

    def middle(member: np.ndarray) -> float | None:
        return float(np.median(probability[member])) if member.any() else None

    by_width, by_count = reliability(probability, right), equal_count(probability, right)
    of_truth = probabilities[np.arange(len(truth)), truth]
    return {"ece": expected_calibration_error(probability, right, BINS), "bins": by_width,
            "ece_equal_count": _gap(by_count), "bins_equal_count": by_count,
            "mean_probability": float(probability.mean()), "share_right": float(right.mean()),
            "probability_minus_right": mean_interval(probability - right, counts),
            "brier": mean_interval(((probabilities - np.eye(N)[truth]) ** 2).sum(1), counts),
            "log_loss": mean_interval(-np.log(np.maximum(of_truth, 1e-12)), counts),
            "median_probability": middle(np.ones(len(right), bool)), "median_probability_right": middle(right),
            "median_probability_wrong": middle(~right), "separation": separation_interval(probability, right, counts)}


def seeds_part(agree: np.ndarray, right: np.ndarray) -> dict:
    """Whether the seeds name one view, against whether the answer is right."""
    agree, right = np.asarray(agree, bool), np.asarray(right, bool)
    return {"recordings_where_the_seeds_differ": int((~agree).sum()), "of_them_wrong": int((~agree & ~right).sum()),
            "wrong_where_the_seeds_agree": int((agree & ~right).sum())}


def word(difference: Mapping, p_two_sided: float) -> str:
    """The plan's word for `classify`'s rule. It needs both: the paired interval of the accuracy wholly on one side of
    zero, and an exact test of the recordings that only one rule reads right with p below 0.05."""
    interval = difference["interval"]
    if interval is None or not p_two_sided < P_BELOW:
        return NOT_SHOWN
    return MORE if interval[0] > 0 else LESS if interval[1] < 0 else NOT_SHOWN


def described(decided: Mapping[str, Classification], natives: tuple[Sequence, ...], draws: int, seed: int) -> dict:
    """Everything that follows from what `classify` returns and the reference views alone."""
    views = given_views(decided, natives)
    truth = np.array([int(s.labels[0]) for s in natives])
    right = np.array([views[s.id] for s in natives]) == truth
    unit = units(natives)
    probabilities = np.stack([decided[s.id].probabilities for s in natives])
    agree = np.array([decided[s.id].seeds_agree for s in natives])
    return {"recordings": len(natives), "errors": int((~right).sum()), "units_drawn": int(unit.max()) + 1,
            "units_with_a_wrong_recording": len(set(unit[~right].tolist())), "confusion": confusion(views, natives),
            "probability": probability_part(probabilities, truth, unit, draws, seed), "seeds": seeds_part(agree, right)}


# --------------------------------------------------------------------------------------------------- the trial
def trial(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], report: object) -> dict:
    """The same functions on the development halves. The development report holds the numbers of this rule for the
    published system: the wrong recordings, the confusion table, the calibration error and the median probability.
    They must come out: the counts exactly, the two numbers made of probabilities within 0.00005. Stops when one
    differs."""
    if not isinstance(report, Mapping) or TRIAL_SYSTEM not in report.get("systems", {}):
        raise SystemExit("the report is not the development report of the published system")
    out, problems = {}, []
    for half in benchmark.HALVES:
        natives = of_kind(halves[half], "native")
        got, known = described(decisions(stored, natives), natives, draws=1, seed=SEED), report["systems"][TRIAL_SYSTEM][half]["native"]
        out[half] = {"recordings": got["recordings"], "errors": got["errors"], "confusion": got["confusion"],
                     "ece": got["probability"]["ece"], "median_probability": got["probability"]["median_probability"]}
        problems += [f"{half}: {key}" for key in ("recordings", "errors", "confusion") if out[half][key] != known[key]]
        problems += [f"{half}: {key}" for key, there in (("ece", "ece"), ("median_probability", "median_confidence"))
                     if not abs(out[half][key] - known[there]) <= CLOSE]  # a number that is not finite does not pass
    if problems:
        raise SystemExit("the development report's numbers for this rule do not come out: " + "; ".join(problems))
    return out


TRIAL_NUMBERS = ("recordings", "errors", "confusion", "ece", "median_probability")


def trial_held(content: object) -> bool:
    """Whether a file is the output of a trial that held: on the development benchmark, against the published
    system's row of the report, with this code's tolerance, the digests of its two inputs and every number of both
    halves. The trial command writes no file when a number does not come out."""
    if not isinstance(content, Mapping) or content.get("benchmark") != "development" or content.get("held_to") != TRIAL_SYSTEM:
        return False
    digests, numbers = content.get("sha256"), content.get("numbers_reproduced")
    if content.get("close") != CLOSE or not isinstance(digests, Mapping) or not isinstance(numbers, Mapping) or set(numbers) != set(benchmark.HALVES):
        return False
    if any(not isinstance(digests.get(key), str) or len(digests[key]) != 64 or set(digests[key]) - set("0123456789abcdef") for key in ("logits", "report")):
        return False
    return all(isinstance(half, Mapping) and all(key in half for key in TRIAL_NUMBERS) and isinstance(half["recordings"], int) and half["recordings"] > 0
               for half in numbers.values())


# ---------------------------------------------------------------------------------------------------- the read
def read(stored: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...], first_read: object, draws: int = DRAWS, seed: int = SEED) -> dict:
    """The read: what `classify` returns on the single recordings of the test benchmark, beside the majority rule.

    The published system's rows come first, from the stored logits. They must give the published numbers and the rows
    of the comparison's first read; `classify`'s rule is applied to no recording before that."""
    natives = of_kind(sequences, "native")
    members = tuple(range(next(iter(stored.values())).shape[1]))
    published = comparison.decoded(member_probabilities(dict(stored), sequences, members), B3)
    result: dict = {"draws": draws, "seed": seed, "published_numbers_reproduced": followup.published_numbers(published, sequences)}
    held = [Row(ROUTER, ROUTER, "A", published), followup.perfect_row(sequences, 5)]
    result["first_read_rows_reproduced"] = followup.held_to_first_read(
        first_read, followup._reported(held, comparison.score_rows(held, sequences, draws, seed)), draws, seed)
    decided = decisions(stored, natives)
    row = as_row(given_views(decided, natives), sequences, published)
    scored = comparison.score_rows([held[0], row], sequences, draws, seed)  # one call: the same draws for both rules
    by_rule = {MAJORITY: scored[ROUTER]["recordings"], CLASSIFY: scored[CLASSIFY]["recordings"]}
    result["rules"] = {name: comparison.report(numbers) for name, numbers in by_rule.items()}
    del result["rules"][CLASSIFY]["cuts_per_recording"]  # a rule that gives one view to a recording has no label change inside it
    result["confusion_majority"] = confusion(majority_views(published, natives), natives)
    result["classify"] = described(decided, natives, draws, seed)
    result["classify_minus_majority"] = {name: comparison.difference(by_rule[CLASSIFY], by_rule[MAJORITY], name, DESCRIBED) for name in COMPARED}
    only = comparison.discordant(natives, row, held[0])
    p = sign_test(only["only_first_right"], only["only_second_right"])
    result["recordings_only_one_rule_reads_right"] = {"only_classify": only["only_first_right"], "only_the_majority": only["only_second_right"],
                                                      "p_two_sided": p}
    result["word"] = word(result["classify_minus_majority"]["accuracy"], p)
    return result


# ----------------------------------------------------------------------------------------------------- commands
def trial_command(args: argparse.Namespace) -> None:
    halves = {half: comparison._scored_kinds(sequences) for half, sequences in benchmark.load().items()}
    stored = comparison._logits(args.logits, tuple(s for half in benchmark.HALVES for s in halves[half]))
    held = trial(stored, halves, json.loads(args.report.read_text()))
    content = {"benchmark": "development", "sha256": {"logits": comparison.file_sha256(args.logits), "report": comparison.file_sha256(args.report)},
               "held_to": TRIAL_SYSTEM, "close": CLOSE, "numbers_reproduced": held}
    if args.out is not None:
        comparison._write(content, args.out)
    print(json.dumps(comparison._clean(content), indent=1))


def read_command(args: argparse.Namespace) -> None:
    """The one read of the test benchmark for `classify`, with everything the plan fixes or not at all."""
    if args.out is None:
        raise SystemExit("the read needs --out")
    missing = [str(path) for path in (args.logits, args.earlier, args.trial) if not path.is_file()]
    if missing:
        raise SystemExit("the test benchmark is not read: not a file: " + ", ".join(missing))
    digests ={"logits": comparison.file_sha256(args.logits), "first_read": comparison.file_sha256(args.earlier), "trial": comparison.file_sha256(args.trial)}
    problems = []
    if args.draws != DRAWS:
        problems.append(f"the plan fixes {DRAWS} draws")
    if digests["logits"] != comparison.TEST_LOGITS_SHA256:
        problems.append("the logits file is not the one the plan names")
    if digests["first_read"] != comparison.FIRST_READ_SHA256:
        problems.append("--earlier must be the output of the comparison's first read, unchanged")
    try:
        held = trial_held(json.loads(args.trial.read_text()))
    except ValueError:
        held = False
    if not held:
        problems.append("--trial must be the output of a trial on development data that held")
    if READ_SHA256 is None and args.reproduce is not None:
        problems.append("there is no read of it to reproduce yet")
    if READ_SHA256 is not None and (args.reproduce is None or not args.reproduce.is_file() or comparison.file_sha256(args.reproduce) != READ_SHA256):
        problems.append("it has been read for this question, and a later run must be given that read's output, unchanged (--reproduce), "
                        "and can only reproduce it")
    if problems:
        raise SystemExit("the test benchmark is not read: " + "; ".join(problems))
    sequences = comparison._scored_kinds(benchmark.load_test())
    stored = comparison._logits(args.logits, sequences)
    result = {"benchmark": "test", "sequences": len(sequences), "sha256": digests,
              **read(stored, sequences, json.loads(args.earlier.read_text()), args.draws, SEED)}
    if args.reproduce is not None and json.loads(args.reproduce.read_text()) != followup._plain(result):
        raise SystemExit("the read does not come out again; nothing is written")
    comparison._write(result, args.out)


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.classify_read", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    p = commands.add_parser("trial", help="the same functions on the development halves, held to the development report; the test benchmark is not read")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    p.add_argument("--report", type=Path, required=True, help="the development report (docs/evaluation/development_report.json)")
    p.add_argument("--out", type=Path, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=trial_command)
    p = commands.add_parser("read", help="the one read of the test benchmark for what classify returns (docs/EVALUATION.md)")
    p.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the test benchmark")
    p.add_argument("--earlier", type=Path, required=True, help="the output of the comparison's first read of the test benchmark (docs/comparison/test.json)")
    p.add_argument("--trial", type=Path, required=True, help="the output of the trial command: the read does not start unless the trial held")
    p.add_argument("--reproduce", type=Path, help="the output of this read, once it exists: the result must equal it, or nothing is written")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--out", type=Path, help="JSON file to write; an existing file is never replaced")
    p.set_defaults(run=read_command)
    args = top.parse_args(argv)
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    args.run(args)
    if args.out is not None:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
