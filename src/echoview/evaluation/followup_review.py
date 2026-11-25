"""Numbers that follow from the output of the read of docs/FOLLOWUP.md, section 9, and from nothing else.

  python -m echoview.evaluation.followup_review docs/followup/confirmation.json [--out docs/followup/confirmation_review.json]

None of them was planned. They were added on 2026-11-25 after a review of the result, to say how far four of its
statements carry. None reads a recording, a label or a model output: the input is the published file with its pooled
numbers, and the command takes no other file. The plan's own code is `followup.py`; this module does not change a
word of the result.

  cut distance with unmatched changes   the plan's distance is a mean over the view changes that have a cut within 1 s.
                                        Here a view change without one counts as 1 s, so every view change counts.
  the recordings that only one reads    an exact two-sided test of the single recordings that only the arm or only the
  right                                 published system reads right (a sign test: each is as likely under no difference).
  both units at once                    the plan takes the interval of (a) over streams and over examinations and joins
                                        the two. Streams and examinations cross, so neither holds all the variation.
                                        Here the half-widths of the two intervals are added in quadrature around the
                                        value. That adds the two variances and counts what they share twice, so the
                                        result is wider than either interval by construction.
  above the floor of the grid           an arm's distance minus that of a perfect system read on the same grid: once
                                        for the matched cuts, as the plan's distance, and once with every view change
                                        counted. A perfect system has a cut for every view change, an arm has not.
  the intervals of the descriptions     how many differences of the output carry a 95% interval and how many of those
                                        exclude zero. None is corrected for their number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from . import comparison, followup
from .scorecard import HZ, NEAR

RADIUS_S = NEAR / HZ  # a cut matches a view change at most this far away (scorecard.NEAR samples)
PUBLISHED = followup.ROUTER
FLOOR = {PUBLISHED: "Perfect on the 0.5 s grid", "every 0.2 s": "Perfect on the 0.2 s grid", "every 0.1 s": None}  # None: read for every sample, 0 s
NOTE = "not planned: added after a review of the result, from the read's output alone (docs/FOLLOWUP.md, section 9)"


def unmatched_at_the_radius(multi: Mapping) -> float:
    """The mean distance over all view changes when one without a matched cut counts as the radius of the matching."""
    mean, matched = multi["cut_distance_mean_s"]["value"], multi["cut_distance_mean_s"]["support"]
    share, changes = multi["cut_matched_share"]["value"], multi["cut_matched_share"]["support"]
    if changes <= 0 or abs(matched / changes - share) > 1e-9:
        raise ValueError(f"the counts of matched cuts do not fit each other: {matched} of {changes} against a share of {share}")
    return (matched * (mean or 0.0) + (changes - matched) * RADIUS_S) / changes


def sign_test(only_first: int, only_second: int) -> float:
    """Exact two-sided p of as uneven a split, when each recording that only one reads right is as likely to be either's."""
    if any(type(k) is not int or k < 0 for k in (only_first, only_second)):
        raise ValueError("the test takes two counts of recordings")
    n, k = only_first + only_second, min(only_first, only_second)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)  # whole numbers until the last division


def both_units(decides: Mapping) -> list[float] | None:
    """An interval for a difference when streams and examinations vary at once: around the value, with the half-widths
    of the two units' intervals added in quadrature. It is at the level of those two intervals, and wider than either."""
    if decides.get("streams") is None or decides.get("examinations") is None:
        return None
    half = 0.5 * math.hypot(*(high - low for low, high in (decides["streams"], decides["examinations"])))
    return [decides["value"] - half, decides["value"] + half]


def above_the_floor(rows: Mapping) -> dict[str, dict[str, float]]:
    """A row's mean distance minus that of a perfect system read on the same grid. `matched_cuts` takes the row's mean
    over its matched cuts, as the plan's distance does. `every_view_change` counts a view change without a matched cut
    as the radius of the matching; a perfect system has a cut for every view change, so its mean needs no such count."""
    out = {}
    for name, floor in FLOOR.items():
        if floor and rows[floor]["multi"]["cut_matched_share"]["value"] != 1.0:
            raise ValueError(f"the row '{floor}' does not match every view change, so it is no floor")
        least = rows[floor]["multi"]["cut_distance_mean_s"]["value"] if floor else 0.0
        out[name] = {"matched_cuts": rows[name]["multi"]["cut_distance_mean_s"]["value"] - least,
                     "every_view_change": unmatched_at_the_radius(rows[name]["multi"]) - least}
    return out


def intervals(differences: Mapping, level: float = 95.0) -> dict:
    """How many differences carry an interval at `level`, and how many of those intervals exclude zero."""
    found = []

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            if "value" in node and "interval" in node:
                if node["interval"] is not None and node.get("level", level) == level:
                    found.append(node["interval"][0] > 0 or node["interval"][1] < 0)
                return
            for child in node.values():
                walk(child)

    walk(differences)
    return {"differences_with_an_interval": len(found), "of_them_exclude_zero": sum(found), "level": level}


def review(result: Mapping) -> dict:
    """Every number of this module for the output of the read."""
    rows, differences, decided = result["rows"], result["differences_from_published"], result["confirmation"]
    exact = {}
    for arm in decided:
        only = differences[arm]["recordings_only_one_reads_right"]
        exact[arm] = {**only, "p_two_sided": sign_test(only["only_the_arm"], only["only_the_published_system"])}
    return {"note": NOTE, "source_sha256": None,
            "cut_distance_with_unmatched_changes_at_1s": {name: unmatched_at_the_radius(rows[name]["multi"]) for name in FLOOR},
            "recordings_exact_test": exact,
            "cut_distance_both_units": {arm: {"value": entry["cut_distance"]["value"], "level": entry["cut_distance"]["level"],
                                              "interval": both_units(entry["cut_distance"])} for arm, entry in decided.items()},
            "cut_distance_above_the_floor_of_its_grid": above_the_floor(rows),
            "descriptions": intervals(differences)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m echoview.evaluation.followup_review", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("confirmation", type=Path, help="docs/followup/confirmation.json, the output of the read")
    parser.add_argument("--out", type=Path, help="write the numbers here as JSON")
    args = parser.parse_args(argv)
    if followup.CONFIRMATION_SHA256 is None:
        raise SystemExit("the code names no output of the read of section 9, so there is nothing to compute from")
    if not args.confirmation.is_file():
        raise SystemExit(f"{args.confirmation} is not a file")
    raw = args.confirmation.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != followup.CONFIRMATION_SHA256:
        raise SystemExit(f"{args.confirmation} is not the output of the read of section 9 (its SHA-256 differs); nothing was computed")
    content = {**review(json.loads(raw)), "source_sha256": digest}
    if args.out:
        try:
            comparison._write(content, args.out)  # whole or not at all, and never in the place of an existing file
        except FileExistsError:
            raise SystemExit(f"{args.out} exists; it is not replaced") from None
    print(json.dumps(content, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
