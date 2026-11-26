"""What part 1 left open, on the development benchmark (docs/ABLATIONS.md, part 3).

  python -m echoview.evaluation.open_points --logits <development logits> --out open.json

No model runs: the input is the window logits of the published system for both development halves, the file of
part 1. Five questions:

  thresholds above the grid     the released decoder at thresholds above the highest of its grid, and what the rule
                                chooses when it may take them
  the halves exchanged          the rule on the selection half, with every setting it weighs there
  the view test                 how many peaks the decoder's view test drops, and whether one of them changes a sample
  the other decoders' numbers   each other decoder at every value of its one number, on both halves
  lower temperatures            temperatures below the lowest of part 1, and the decoder under the fitted temperature

Everything that is chosen is chosen on the calibration half and reported on the selection half. The test benchmark is
not read, and nothing here changes the published system.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .. import decoder
from ..reference import B3
from . import ablations, benchmark, calibrate, comparison, metrics
from .ablations import CANDIDATE, NOT_SUPPORTED
from .benchmark import Sequence, of_kind
from .comparison import DRAWS, ROUTER, SEED, Row
from .run import member_probabilities

HIGHER = (0.9995, 0.9999, 1.0)  # quantiles above the highest of the rule's grid; 1.0 is the largest score inside a single recording
LOWER = (0.05, 0.10, 0.15, 0.20, 0.25)  # temperatures below the lowest of part 1
TEMPERATURES = LOWER + ablations.TEMPERATURES  # 0.05 to 2.00
STAYS = "the published setting stays"
WITH_HIGHER = "the rule with higher thresholds"
KEPT_SETTING, BY_THE_RULE = "the temperature, the published setting", "the temperature, the setting by the rule"
KINDS = ("native", "pairs", "multi")


def ensemble_probabilities(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], temperature: float = 1.0) -> dict:
    """Per-sample probabilities of the whole ensemble for both halves, with every logit divided by the temperature. At a
    temperature of 1 the logits are passed on as they are, so the numbers are those of part 1."""
    members = tuple(range(next(iter(stored.values())).shape[1]))
    logits = dict(stored) if temperature == 1.0 else {key: np.asarray(value, np.float64) / temperature for key, value in stored.items()}
    return {half: member_probabilities(logits, sequences, members) for half, sequences in halves.items()}


def rule_numbers(labels: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...]) -> dict:
    """The numbers the rule weighs, for decoded streams."""
    result = metrics.evaluate(dict(labels), sequences)
    return {key: result[key] for key in calibrate.REPORTED}


def is_published(setting: Mapping) -> bool:
    """The published setting, known by its three numbers. A setting that part 3 adds is never the published one."""
    return (not setting.get("added", False) and all(setting[key] == B3[key] for key in ("half_window", "nms_radius"))
            and abs(setting["threshold"] - B3["threshold"]) < comparison.THRESHOLD_CLOSE)


def by_the_rule(table: list[dict], half: str) -> dict:
    """The project's rule over a table of settings, read on one half: the order of the setting it chooses and how many
    settings keep the two floors. When none keeps them they are dropped, as everywhere in part 3: the setting with the
    highest segment F1@50 on the multi streams takes their place, the earlier of equals, and `floors_dropped` says so."""
    found = calibrate.select_setting([{"order": entry["order"], **entry[half]} for entry in table])
    if found["chosen"] is None:
        chosen = max(table, key=lambda entry: (entry[half]["segment_f1_50_multi"], -entry["order"]))["order"]
    else:
        chosen = found["chosen"]["order"]
    return {"status": found["status"], "n_feasible": found["n_feasible"], "n_settings": found["n_settings"], "floors_dropped": found["chosen"] is None,
            "chosen": chosen}


# ------------------------------------------------------------------------------------- thresholds above the grid
def extended_grid(native_probabilities: list[np.ndarray]) -> list[dict]:
    """The rule's grid with the higher quantiles after its own, for every half window, in the rule's order."""
    out = []
    for half_window in calibrate.HALF_WINDOWS:
        scores = np.concatenate([decoder.boundary_scores(p, half_window)[1:] for p in native_probabilities])
        for quantile in (*calibrate.QUANTILES, *HIGHER):
            out.append({"half_window": int(half_window), "threshold": float(np.quantile(scores, float(quantile))), "nms_radius": calibrate.NMS_RADIUS,
                        "threshold_quantile": float(quantile), "added": quantile in HIGHER})
    return out


def settings_table(grid: list[dict], probabilities: Mapping[str, Mapping[str, np.ndarray]], halves: Mapping[str, tuple[Sequence, ...]]) -> list[dict]:
    """Every setting of a grid with the numbers of the rule on both halves."""
    return [{"order": order, **setting, **{half: rule_numbers(calibrate.decode_all(setting, probabilities[half]), sequences)
                                           for half, sequences in halves.items()}} for order, setting in enumerate(grid)]


def higher_thresholds(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int, seed: int) -> dict:
    """The released decoder at thresholds above its grid. The rule chooses on the calibration half over the grid and the
    added settings together, and the selection half reports its choice against the published setting."""
    probabilities = ensemble_probabilities(stored, halves)
    table = settings_table(extended_grid([probabilities["calibration"][s.id] for s in of_kind(halves["calibration"], "native")]), probabilities, halves)
    rule = by_the_rule(table, "calibration")
    chosen = table[rule["chosen"]]
    rows = [Row(ROUTER, ROUTER, "A", comparison.decoded(probabilities["selection"], B3)),
            Row(WITH_HIGHER, ROUTER, "D", comparison.decoded(probabilities["selection"], chosen))]
    scored = comparison.score_rows(rows, halves["selection"], draws, seed)
    apart = ablations.minus(scored[WITH_HIGHER], scored[ROUTER])
    interval = apart["multi"]["segment_f1_50"]["interval"]
    if is_published(chosen):
        word = STAYS
    elif interval is not None and interval[0] > 0 and apart["recordings"]["accuracy"]["value"] >= 0:
        word = CANDIDATE
    else:
        word = NOT_SUPPORTED
    published = metrics.evaluate(comparison.decoded(probabilities["selection"], B3), halves["selection"])["segment_f1_50_multi"]
    added = [entry["selection"]["segment_f1_50_multi"] for entry in table if entry["added"]]
    return {"settings": table,
            "rule": {**rule, "is_the_published_setting": is_published(chosen), "is_at_the_highest_quantile": chosen["threshold_quantile"] == max(HIGHER)},
            "selection": {"published": ablations.numbers(scored[ROUTER]), "chosen": ablations.numbers(scored[WITH_HIGHER]), "chosen_minus_published": apart},
            "added_on_selection": {"settings": len(added), "above_the_published": int(sum(value > published for value in added)), "highest": float(max(added))},
            "word": word}


# ------------------------------------------------------------------------------------------ the halves exchanged
def halves_exchanged(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]]) -> dict:
    """The rule the other way round: the grid's thresholds come from the selection half and the rule chooses there. Every
    setting is given on both halves, and how many keep the floors on the half that chooses."""
    probabilities = ensemble_probabilities(stored, halves)
    table = settings_table(calibrate.grid([probabilities["selection"][s.id] for s in of_kind(halves["selection"], "native")]), probabilities, halves)
    for entry in table:
        entry["feasible_on_selection"] = bool(entry["selection"]["native_macro_f1"] >= calibrate.NATIVE_FLOOR
                                              and (entry["selection"]["boundary_recall_multi"] or 0.0) >= calibrate.RECALL_FLOOR)
    highest = max(table, key=lambda entry: (entry["selection"]["segment_f1_50_multi"], -entry["order"]))
    return {"settings": table, "rule": by_the_rule(table, "selection"), "highest_on_selection_whatever_the_floors": highest["order"]}


# -------------------------------------------------------------------------------------------------- the view test
def view_test(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]]) -> dict:
    """What the decoder's view test does at the published setting: the peaks of the score, how many the test drops,
    and in how many streams and samples the decoder without the test gives another view."""
    probabilities = ensemble_probabilities(stored, halves)
    half_window, threshold, radius = int(B3["half_window"]), float(B3["threshold"]), int(B3["nms_radius"])
    out: dict = {}
    for half, sequences in halves.items():
        out[half] = {}
        for kind in KINDS:
            count = {"streams": 0, "samples": 0, "peaks": 0, "kept": 0, "dropped": 0, "streams_with_a_dropped_peak": 0, "streams_that_change": 0,
                     "samples_that_change": 0}
            for s in of_kind(sequences, kind):
                prob = probabilities[half][s.id]
                scores = decoder.boundary_scores(prob, half_window)
                peaks = decoder.non_max_suppression(scores, threshold, radius)
                peaks = peaks[peaks > 0]
                kept = decoder.detect_boundaries(prob, half_window, threshold, radius, scores)
                changed = int((decoder.decode(prob, half_window, threshold, radius, scores) != ablations.without_the_view_test(prob, B3)).sum())
                count["streams"] += 1
                count["samples"] += len(prob)
                count["peaks"] += int(peaks.size)
                count["kept"] += int(kept.size)
                count["dropped"] += int(peaks.size - kept.size)
                count["streams_with_a_dropped_peak"] += int(peaks.size > kept.size)
                count["streams_that_change"] += int(changed > 0)
                count["samples_that_change"] += changed
            out[half][kind] = count
    return out


# ------------------------------------------------------------------------------------ the other decoders' numbers
def numbers_of_the_other_decoders(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]]) -> dict:
    """Each other decoder at every value of its one number, with the numbers of the rule on both halves, and how the
    selection half's segment F1@50 spreads around the value that the rule chose."""
    probabilities = ensemble_probabilities(stored, halves)
    out = {}
    for name, values in ablations.GRIDS.items():
        table = [{"order": order, "value": value, **{half: rule_numbers(ablations.decode_with(name, value, probabilities[half]), sequences)
                                                     for half, sequences in halves.items()}} for order, value in enumerate(values)]
        rule = by_the_rule(table, "calibration")
        chosen = table[rule["chosen"]]
        scores = np.array([entry["selection"]["segment_f1_50_multi"] for entry in table])
        at = chosen["selection"]["segment_f1_50_multi"]
        out[name] = {"values": table, "rule": rule, "chosen": chosen["value"],
                     "selection_segment_f1_50_multi": {"of_the_chosen": float(at), "lowest": float(scores.min()), "highest": float(scores.max()),
                                                       "within_0.01_of_the_chosen": int((np.abs(scores - at) <= 0.01).sum()),
                                                       "within_0.02_of_the_chosen": int((np.abs(scores - at) <= 0.02).sum()),
                                                       "above_the_chosen": int((scores > at).sum()), "values": len(scores)}}
    return out


# --------------------------------------------------------------------------------------------- lower temperatures
def lower_temperatures(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int, seed: int) -> dict:
    """One temperature as in part 1, by part 1's own function, with lower values allowed: fitted on the single
    recordings of the calibration half and reported on those of the selection half. Then the streams of the selection
    half under that temperature, with the published decoder setting and with the setting that the rule gives the
    changed probabilities on the calibration half (its own grid of 40; the floors are dropped when none keeps them)."""
    out = ablations.one_temperature(stored, halves, draws, seed, TEMPERATURES)
    chosen = out["chosen"]
    out["is_the_lowest_tried"] = chosen == min(TEMPERATURES)
    # the decoder reads the same probabilities: the streams under the temperature
    plain, changed = ensemble_probabilities(stored, halves), ensemble_probabilities(stored, halves, chosen)
    calibration = halves["calibration"]
    setting = comparison.choose(changed["calibration"], calibration, drop_floors=True)
    thresholds = [entry["threshold"] for entry in calibrate.grid([changed["calibration"][s.id] for s in of_kind(calibration, "native")])]
    rows = [Row(ROUTER, ROUTER, "A", comparison.decoded(plain["selection"], B3)),
            Row(KEPT_SETTING, ROUTER, "D", comparison.decoded(changed["selection"], B3)),
            Row(BY_THE_RULE, ROUTER, "D", comparison.decoded(changed["selection"], setting["params"]))]
    scored = comparison.score_rows(rows, halves["selection"], draws, seed)
    out["streams"] = {"published": ablations.numbers(scored[ROUTER]),
                      "setting_by_the_rule": {"status": setting["status"], "n_feasible": setting["n_feasible"], "floors_dropped": setting["floors_dropped"],
                                              **setting["params"]},
                      # sharp probabilities can make the thresholds of the rule's grid fall together: the grid then offers fewer settings than 40
                      "grid_of_the_rule": {"settings": len(thresholds), "distinct_thresholds": len({round(t, 12) for t in thresholds}),
                                           "thresholds_at_zero": int(sum(t <= 0.0 for t in thresholds))},
                      **{name: {"selection": ablations.numbers(scored[name]), "minus_published": ablations.minus(scored[name], scored[ROUTER])}
                         for name in (KEPT_SETTING, BY_THE_RULE)}}
    return out


def study(stored: Mapping[str, np.ndarray], halves: Mapping[str, tuple[Sequence, ...]], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The five questions of part 3."""
    return {"benchmark": "development", "draws": draws, "seed": seed,
            "sequences": {half: len(sequences) for half, sequences in halves.items()},
            "higher_thresholds": higher_thresholds(stored, halves, draws, seed), "halves_exchanged": halves_exchanged(stored, halves),
            "view_test": view_test(stored, halves), "other_decoders": numbers_of_the_other_decoders(stored, halves),
            "temperature": lower_temperatures(stored, halves, draws, seed)}


# ----------------------------------------------------------------------------------------------------- command
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m echoview.evaluation.open_points", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logits", type=Path, required=True, help="the window logits of the published system for the development benchmark")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    halves = ablations.development_halves()
    stored = comparison._logits(args.logits, tuple(s for half in benchmark.HALVES for s in halves[half]))
    f1 = ablations.held_to_published(stored, halves["selection"])
    result = {"sha256": {"logits": comparison.file_sha256(args.logits)}, "published_selection_f1": f1, **study(stored, halves, args.draws, SEED)}
    comparison._write(result, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
