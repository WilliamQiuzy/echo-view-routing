"""The results of part 4 of docs/ABLATIONS.md (section 13), written from its output file.

  python tools/write_arms_results.py docs/ablations/arms.json docs/ablations/forward.json

The command prints the section. Every number of the text comes from the file, and no number is rounded by hand. The
sentences that say something without a number (an interval on one side of zero, a count, an order) are asked of the
file first, and nothing is printed when the file says otherwise. The checks are assertions, so the tool does not run
under `python -O`. Two sentences need the package: that every model is the run the plan names, option for option, and
that no weight file is a published one. Some sentences cannot be asked of a file: when the run happened and what the
server did (from the logs of the run), what the report checked before it wrote its file (from its code), and what the
plan says. An option that is added to the trainer later must leave the identity of a run without it as it was, as
`--mirror` does, or the check of the identities fails. tests/test_docs_arms.py holds the document to what this prints.
"""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR = "published", "stage 2", "again", "without input changes", "with a mirror"
SYSTEMS = (PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR)
ARMS = (AGAIN, WITHOUT, MIRROR)
CLEAN, MIRRORED = "unchanged", "mirrored"
FAMILY = ("noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5")
PAIRS = ((AGAIN, WITHOUT), (WITHOUT, STAGE2), (AGAIN, STAGE2), (MIRROR, AGAIN), (MIRROR, PUBLISHED), (PUBLISHED, AGAIN), (PUBLISHED, STAGE2))
OWN, NONE = "the setting by the rule", "no decoder"
MOSTLY_CHANGES, NOT_TOLD = "mostly the input changes", "not told apart"
RECORDINGS = 284  # the single recordings of the selection half
ROOT = Path(__file__).resolve().parents[1]
DESCRIBED, DECIDES = 95.0, 99.0  # the level of the intervals that describe, and of those that decide a sentence of the first question
NOISE = Decimal("1e-12")  # a number is taken at twelve decimals: what a sum of binary numbers is off by lies far below that
NAMED = {PUBLISHED: "the published system", STAGE2: "the stage-2 ensemble", AGAIN: "\"again\"", WITHOUT: "\"without input changes\"",
         MIRROR: "\"with a mirror\""}  # a system as a sentence names it
NUMERAL = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
NUMBERS = (  # the name, the part, the key, the digits, the scale: every number of section 5
    ("Segment F1 at 50%, multi streams", "Segment F1, multi", "multi", "segment_f1_50", 3, 1.0),
    ("Edit", "Edit", "multi", "edit", 1, 1.0),
    ("Wrong samples, percentage points", "Wrong samples, points", "multi", "wrong_samples", 1, 100.0),
    ("Cut precision", "Cut precision", "multi", "cut_precision", 3, 1.0),
    ("Cut recall", "Cut recall", "multi", "cut_recall", 3, 1.0),
    ("Mean distance of matched cuts, seconds", "Cut distance, s", "multi", "cut_distance_mean_s", 3, 1.0),
    ("Cuts at joins of the same view, percentage points", "Cuts at same-view joins, points", "multi", "same_view_cut_rate", 1, 100.0),
    ("Segments that match nothing", "Segments that match nothing", "multi", "extra_segments", 0, 1.0),
    ("Segments missed", "Segments missed", "multi", "missed_segments", 0, 1.0),
    ("Segment F1 at 50%, pair streams", "Segment F1, pairs", "pairs", "segment_f1_50", 3, 1.0),
    ("Accuracy, single recordings", "Accuracy", "recordings", "accuracy", 3, 1.0),
    ("Macro-F1, single recordings", "Macro-F1", "recordings", "macro_f1", 3, 1.0),
    ("A4C read as A5C, recordings", "A4C as A5C", "recordings", "a4c_as_a5c", 0, 1.0),
    ("A5C read as A4C, recordings", "A5C as A4C", "recordings", "a5c_as_a4c", 0, 1.0),
    ("Label changes per recording", "Label changes per recording", "recordings", "cuts_per_recording", 3, 1.0),
)


# ------------------------------------------------------------------------------------------------- how a number is shown
def rounded(x: float, digits: int, sign: bool = False, scale: float = 1.0) -> str:
    """A number at the digits shown, with a tie rounded away from zero as in ordinary rounding (Python's own formatting
    rounds a tie of the binary number to even). A scale is applied to the decimal number, not to the binary one. The
    number is taken at twelve decimals first, so that a tie which the file stores a hair off itself is the tie: 15,352
    of 16,000 is stored as 0.9594999999999999."""
    step = Decimal(1).scaleb(-digits)
    exact = (Decimal(repr(float(x))) * Decimal(repr(float(scale)))).quantize(NOISE, rounding=ROUND_HALF_EVEN)
    value = exact.quantize(step, rounding=ROUND_HALF_UP)
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def f3(x: float) -> str:
    return rounded(x, 3)


def at(entry: dict, level: float) -> dict:
    """A difference or a share: its interval is at the level the text gives it, and the value lies inside it."""
    assert entry["level"] == level, (entry["level"], level)
    assert entry["interval"] is None or entry["interval"][0] <= entry["value"] <= entry["interval"][1]
    return entry


def signed(entry: dict, digits: int = 3, scale: float = 1.0, level: float = DESCRIBED) -> str:
    """A difference with its sign, and its interval where it has one. The interval is at 95% unless another level is named."""
    at(entry, level)
    if entry.get("interval") is None:
        return rounded(entry["value"], digits, True, scale)
    low, high = entry["interval"]
    return f"{rounded(entry['value'], digits, True, scale)} [{rounded(low, digits, True, scale)}, {rounded(high, digits, True, scale)}]"


def plain(entry: dict, level: float | None = None) -> str:
    """A value with its interval, without signs. A number of one system names no level; a share does."""
    assert entry.get("level") == level, (entry.get("level"), level)
    return f"{f3(entry['value'])} [{f3(entry['interval'][0])}, {f3(entry['interval'][1])}]"


def below(entry: dict) -> bool:
    return entry.get("interval") is not None and entry["interval"][1] < 0


def above(entry: dict) -> bool:
    return entry.get("interval") is not None and entry["interval"][0] > 0


def holds_zero(entry: dict) -> bool:
    return entry["interval"][0] <= 0 <= entry["interval"][1]


def around_zero(entry: dict) -> bool:
    """Zero lies inside the interval and at neither end: neither a loss nor a gain is excluded."""
    return entry["interval"][0] < 0 < entry["interval"][1]


def whole(x: float) -> int:
    assert abs(x - round(x)) < 1e-6, x
    return round(x)


def listed(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def numeral(count: int) -> str:
    assert 0 <= count < len(NUMERAL), count
    return NUMERAL[count]


def span(values, digits: int = 3, sign: bool = False) -> str:
    values = list(values)
    return f"{rounded(min(values), digits, sign)} to {rounded(max(values), digits, sign)}"


# ------------------------------------------------------------------------------------------------- where a number stands
def conditions(d: dict) -> list[str]:
    return list(d["systems"][PUBLISHED]["conditions"])


def number(d: dict, system: str, name: str = CLEAN, part: str = "multi", key: str = "segment_f1_50") -> dict:
    return d["systems"][system]["conditions"][name]["selection"][part][key]


def value(d: dict, system: str, name: str = CLEAN, part: str = "multi", key: str = "segment_f1_50") -> float:
    return number(d, system, name, part, key)["value"]


def right(d: dict, system: str, name: str = CLEAN) -> int:
    return whole(RECORDINGS * value(d, system, name, "recordings", "accuracy"))


def wrong(d: dict, system: str, name: str = CLEAN) -> int:
    return d["systems"][system]["conditions"][name]["classify"]["errors"]


def apart(d: dict, first: str, second: str, name: str = CLEAN, part: str = "multi", key: str = "segment_f1_50") -> dict:
    return at(d["differences"][f"{first} minus {second}"][name][part][key], DESCRIBED)


def own_loss(d: dict, system: str, name: str) -> dict:
    return at(d["systems"][system]["conditions"][name]["minus_unchanged"]["multi"]["segment_f1_50"], DESCRIBED)


def seed_apart(d: dict, system: str, k: int) -> dict:
    """A seed of a system minus the seed of the same number of the arm "again", on the unchanged frames."""
    return at(d["seeds"]["minus_again"][system][f"seed {k}"]["multi"]["segment_f1_50"], DESCRIBED)


def one_sided(d: dict, first: str, second: str) -> tuple[list[str], list[str]]:
    """The conditions under which the interval of a difference lies wholly above zero, and wholly below."""
    names = conditions(d)
    return [n for n in names if above(apart(d, first, second, n))], [n for n in names if below(apart(d, first, second, n))]


# ------------------------------------------------------------------------------------------- what the file must hold
def consistent(d: dict, part2: dict) -> None:
    """The file holds what the text takes it to hold: the conditions of part 2 in their order for every system and every
    difference, the numbers of section 5 and no other, and every difference is the difference of the two values."""
    names = conditions(d)
    keys = {part: {key for _, _, held, key, _, _ in NUMBERS if held == part} for part in ("multi", "pairs", "recordings")}
    assert names == list(part2["systems"][PUBLISHED]["conditions"]) and len(names) == 27 and names[0] == CLEAN and tuple(d["systems"]) == SYSTEMS
    for system in SYSTEMS:
        held = d["systems"][system]["conditions"]
        assert list(held) == names and "minus_unchanged" not in held[CLEAN]
        for name in names:
            numbers = held[name]["selection"]
            assert {part: set(numbers[part]) for part in numbers} == keys and numbers["recordings"]["accuracy"]["support"] == RECORDINGS
            for part in keys if name != CLEAN else ():
                for key in keys[part]:
                    lost = at(held[name]["minus_unchanged"][part][key], DESCRIBED)
                    assert abs(lost["value"] - (numbers[part][key]["value"] - held[CLEAN]["selection"][part][key]["value"])) < 1e-12
        # with the rule's setting and with no decoder a system has its numbers on the unchanged frames, and nothing else
        other = d["systems"][system]["unchanged_with"]
        assert set(other) == {OWN, NONE} and set(other[OWN]) == {"setting", "selection"} and set(other[NONE]) == {"selection"}
        assert all({part: set(other[with_it]["selection"][part]) for part in other[with_it]["selection"]} == keys for with_it in (OWN, NONE))
    assert list(d["differences"]) == [f"{a} minus {b}" for a, b in PAIRS]
    for first, second in PAIRS:
        by = d["differences"][f"{first} minus {second}"]
        assert list(by) == names  # the 27 conditions and nothing else: no difference was scored with another decoder setting
        for name in names:
            assert {part: set(by[name][part]) for part in by[name]} == keys
            for part in keys:
                for key in keys[part]:
                    both = value(d, first, name, part, key) - value(d, second, name, part, key)
                    assert abs(apart(d, first, second, name, part, key)["value"] - both) < 1e-12
    seeds = d["seeds"]
    assert set(seeds) == {"systems", "minus_again"} and set(seeds["minus_again"]) == {PUBLISHED, WITHOUT, MIRROR}
    assert set(seeds["systems"]) == {f"{system} | seed {k}" for system in (PUBLISHED, *ARMS) for k in range(3)}
    assert all({part: set(numbers[part]) for part in numbers} == keys for numbers in seeds["systems"].values())
    for system in (PUBLISHED, WITHOUT, MIRROR):
        for k in range(3):
            assert {part: set(seeds["minus_again"][system][f"seed {k}"][part]) for part in seeds["minus_again"][system][f"seed {k}"]} == keys
            mine, partner = (seeds["systems"][f"{name} | seed {k}"]["multi"]["segment_f1_50"]["value"] for name in (system, AGAIN))
            assert abs(seed_apart(d, system, k)["value"] - (mine - partner)) < 1e-12
    # the first question: every difference and every share is what the values of the systems give
    assert tuple(d["input_changes"]["conditions"]) == FAMILY
    for name, e in d["input_changes"]["conditions"].items():
        again, without, stage2, published = (value(d, system, name) for system in (AGAIN, WITHOUT, STAGE2, PUBLISHED))
        second = e["with_the_published_system"]["share"]
        for entry, given in ((e["again_minus_without"], again - without), (e["share"]["of"], again - stage2), (e["published_minus_again"], published - again),
                             (e["share"], (again - without) / (again - stage2)), (second["of"], published - stage2),
                             (second, (published - without) / (published - stage2))):
            assert abs(at(entry, DECIDES)["value"] - given) < 1e-12
    # the published system and the stage-2 ensemble are the files of part 2: the same machine and version, the same draws
    assert (d["draws"], d["seed"], d["sequences"]) == (part2["draws"], part2["seed"], part2["sequences"])
    assert all(d["systems"][system]["system"] == part2["systems"][system]["system"] for system in (PUBLISHED, STAGE2))


def held_to_the_package(d: dict) -> None:
    """Two sentences that the file alone cannot carry: every model is the run that the plan names for its arm and its
    seed, option for option, and no weight file of an arm is a published one. The names and the margins of this tool
    are those of the code that wrote the file."""
    try:
        from echoview import weights
        from echoview.evaluation import arms, comparison
    except ImportError as error:
        raise SystemExit(f"this tool needs the package it stands in ({error}): pip install -e .") from error
    assert tuple(arms.ARMS) == ARMS and arms.SYSTEMS == SYSTEMS and tuple(arms.DIFFERENCES) == PAIRS and arms.FAMILY == FAMILY
    assert (arms.OWN_LOSS, arms.UNCHANGED, arms.FAMILY_LEVEL, comparison.DESCRIBED) == (0.03, 0.01, DECIDES, DESCRIBED)
    known = {digest for files in weights.WEIGHT_SHA256.values() for digest in files.values()} | set(weights.START_SHA256.values())
    for arm in ARMS:
        about = d["systems"][arm]["system"]
        assert about["weights_sha256"] == [model["sha256"] for model in about["models"]] and not known & set(about["weights_sha256"])
        for model in about["models"]:
            assert model["identity"] == arms.recipe_of(arm, model["seed"], ROOT / arms.RECIPE)


# ------------------------------------------------------------------------------------------------------------ the run
def run_text(d: dict, part2: dict, part2_sha256: str) -> list[str]:
    assert d["sha256"]["part2"] == part2_sha256  # the output of part 2 that the report was held to is the file given here
    assert (d["benchmark"], d["half"], d["sequences"], d["draws"], d["seed"]) == ("development", "selection", 541, 10_000, 20261123)
    assert d["code"] == "07bf68e" and len(conditions(d)) == 27 and conditions(d)[0] == CLEAN and tuple(d["systems"]) == SYSTEMS
    assert tuple(d["differences"]) == tuple(f"{a} minus {b}" for a, b in PAIRS) and tuple(d["input_changes"]["conditions"]) == FAMILY
    about = {arm: d["systems"][arm]["system"] for arm in ARMS}
    versions = {m["torch"] for arm in ARMS for m in about[arm]["models"]} | {d["systems"][s]["system"]["torch"] for s in SYSTEMS}
    assert len(versions) == 1 and {about[arm]["code"] for arm in ARMS} == {d["code"]}
    version = next(iter(versions))
    hashes = [h for arm in ARMS for h in about[arm]["weights_sha256"]]
    assert len(set(hashes)) == 9 and all(about[arm]["set_against_sha256"] == about[AGAIN]["weights_sha256"] for arm in (WITHOUT, MIRROR))
    assert d["unchanged_against_stored"] < 1e-6 and set(d["unchanged_against_own_logits"]) == set(ARMS)
    assert all(x == 0.0 for x in d["unchanged_against_own_logits"].values())
    assert rounded(value(d, PUBLISHED), 4) == "0.9594" and d["sha256"]["logits"] == part2["sha256"]["logits"]
    assert d["decoder"] == {"half_window": 7, "threshold": 0.1737918038235492, "nms_radius": 5}
    # the published system under every condition is, number for number, what part 2 holds
    for name in conditions(d):
        mine, theirs = d["systems"][PUBLISHED]["conditions"][name], part2["systems"][PUBLISHED]["conditions"][name]
        assert all(mine[block] == theirs[block] for block in ("selection", "classify")) and mine.get("minus_unchanged") == theirs.get("minus_unchanged")
    # and the stage-2 ensemble with the rule's setting is the stage-2 ensemble of part 2
    ruled, before = d["systems"][STAGE2]["unchanged_with"][OWN], part2["systems"][STAGE2]
    assert {k: ruled["setting"][k] for k in before["setting"]["params"]} == before["setting"]["params"]
    assert ruled["selection"] == before["conditions"][CLEAN]["selection"]
    return [
        "**Run from 2026-11-27 01:34 UTC to 2026-11-28 04:58 UTC with the code of commit `07bf68e`, which holds the plan of section 12.** The output of the "
        f"report is [`ablations/arms.json`](ablations/arms.json). The nine models were trained on 2026-11-27 from 01:34 to 03:43 UTC, on one L40S with "
        f"PyTorch {version}. The three arms were then run under the conditions on the same server, one arm after the other, until 04:54 UTC on 2026-11-28, "
        "and the report was written at 04:58 UTC. The server is a preemptible one. On 2026-11-27 it stopped answering at about 08:01 UTC and was started "
        "again at 08:09 UTC. It stopped answering again at about 10:12 UTC, was found stopped, and was started again at 02:52 UTC on 2026-11-28. The arms "
        "\"without input changes\" and \"with a mirror\" each went on from the files they had, and a condition whose file existed was not computed again. "
        "No training was cut off. None of the nine models is published, and the test benchmark was not read.",
        "",
        "**The published system and the stage-2 ensemble were not run again.** Their outputs under the 27 conditions are the files that part 2 wrote on "
        "2026-11-26 with the code of commit `fb674f6` (section 9), on the same server and with the same version of PyTorch. The code that runs a system "
        "under a condition is the same in both commits. The report of part 4 scored those files again beside the arms, with the same draws.",
        "",
        "Every number about streams and recordings below is for the selection half. The decoder setting is the published one, half window 7 and threshold "
        "0.174, unless a line names another. Intervals are at 95% unless a line says 99%. They are many, none is corrected for their number, and they hold "
        "nothing of the difference between two trainings. An end printed as +0.000 or -0.000 can be a number just off zero; the file has every digit.",
        "",
        "**The checks of section 12 passed before a number was read,** and the report is not written when one fails.",
        "",
        "- **The published system.** The outputs of its unchanged frames in the files of part 2 are the stored outputs, to less than 0.000001 in every "
        "probability, and they give the published number of the selection half, 0.9594. Scored again here, its numbers under all 27 conditions are those of "
        "part 2 to the last digit.",
        "- **Held to part 2.** The report scores the published system minus the stage-2 ensemble once more with the stage-2 ensemble decoded as part 2 "
        "decoded it, and it stops unless every number under every condition is what part 2 published. It did not stop. The output of part 2 was made from "
        "the stored logits that the report read: the two files name the same checksum. The result names both files by their checksums.",
        "- **The arms.** Each arm's run of the unchanged frames gave the same probabilities as its own window logits. Each of the nine models is the finished "
        "run that section 12 names for its arm and its seed, option for option, started from the published stage-2 model. The nine weight files are nine "
        "different files and none is a published one. The nine runs name one version of PyTorch and the three arms one commit, and the other two arms were "
        "set against the three models of the arm \"again\" that the report was given.",
    ]


# ------------------------------------------------------------------------------------------------------- the training
def model_rows(d: dict) -> list[str]:
    rows = ["| Arm | Seed | Sessions | Loss with its penalty, last 500 updates | Accuracy on its training windows, last 500 updates | "
            "Weights from their start | From the model of \"again\" of the same seed |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        about = d["systems"][arm]["system"]
        for model, distance in zip(about["models"], about["distances"]):
            assert model["seed"] == distance["seed"]
            other = "" if arm == AGAIN else rounded(distance["from_again"], 5)
            rows.append(f"| {arm} | {model['seed']} | {model['sessions']} | {f3(model['training']['loss'])} | {f3(model['training']['accuracy'])} | "
                        f"{rounded(distance['from_its_start'], 5)} | {other} |")
    return rows


def training_text(d: dict) -> list[str]:
    about = {arm: d["systems"][arm]["system"] for arm in ARMS}
    models = {arm: about[arm]["models"] for arm in ARMS}
    assert all([m["seed"] for m in models[arm]] == [0, 1, 2] for arm in ARMS)
    assert all(m["sessions"] == 1 for arm in ARMS for m in models[arm])  # no run was continued
    for arm in ARMS:
        for m in models[arm]:  # 30 lines of 50 updates each: the last ten lines are the last 500 updates
            assert (m["training"]["lines"], m["training"]["last_update"], m["identity"]["log_every"], m["identity"]["updates"]) == (30, 1500, 50, 1500)
            assert m["identity"]["init"] == "stage2" and m["identity"]["penalty"] == 0.15
            assert (m["identity"]["strength"], m["identity"].get("mirror", 0.0)) == {AGAIN: (1.0, 0.0), WITHOUT: (0.0, 0.0), MIRROR: (1.0, 0.5)}[arm]
    loss = {arm: [m["training"]["loss"] for m in models[arm]] for arm in ARMS}
    accuracy = {arm: [m["training"]["accuracy"] for m in models[arm]] for arm in ARMS}
    start = {arm: [e["from_its_start"] for e in about[arm]["distances"]] for arm in ARMS}
    other = {arm: [e["from_again"] for e in about[arm]["distances"]] for arm in (WITHOUT, MIRROR)}
    for k in range(3):  # for every seed: without input changes fits best and moved least, with a mirror fits worst and moved most
        assert loss[WITHOUT][k] < loss[AGAIN][k] < loss[MIRROR][k] and accuracy[WITHOUT][k] > accuracy[AGAIN][k] > accuracy[MIRROR][k]
        assert start[WITHOUT][k] < start[AGAIN][k] < start[MIRROR][k]
        assert other[MIRROR][k] < start[MIRROR][k]  # the arm with a mirror lies nearer to "again" than to its own start
    for arm in ARMS:  # an accuracy over the last ten lines of a log is a share of whole windows
        for m in models[arm]:
            counted = m["training"]["accuracy"] * 10 * m["identity"]["log_every"] * m["identity"]["batch"]
            assert abs(counted - round(counted)) < 1e-6
    return [
        "- **Each of the nine models was trained in one session.** No run went on from a checkpoint.",
        f"- **The arm without input changes fits its training windows best, and the arm with a mirror worst.** Over the last 500 updates the loss is "
        f"{span(loss[WITHOUT])} without input changes, {span(loss[AGAIN])} for \"again\" and {span(loss[MIRROR])} with a mirror, and the accuracy is "
        f"{span(accuracy[WITHOUT])}, {span(accuracy[AGAIN])} and {span(accuracy[MIRROR])}. The order is the same for every seed. These numbers are for the "
        "windows as each arm was fed them.",
        f"- **The weights of the arm without input changes moved least, those of the arm with a mirror most.** As a share of length the models lie "
        f"{span(start[WITHOUT], 5)} from their stage-2 start without input changes, {span(start[AGAIN], 5)} for \"again\" and {span(start[MIRROR], 5)} with a "
        f"mirror, in this order for every seed. Between \"without input changes\" and \"again\" of one seed the distance is {span(other[WITHOUT], 5)}, "
        f"and between \"with a mirror\" and \"again\" {span(other[MIRROR], 5)}: for every seed the arm with a mirror lies nearer to \"again\" than to "
        "its own start.",
    ]


# ----------------------------------------------------------------------------------------------- the unchanged frames
def unchanged_rows(d: dict) -> list[str]:
    rows = ["| System | Segment F1 at 50%, multi streams | Pair streams | Recordings right of 284, majority rule | Wrong by `classify`'s rule | "
            "Setting by the rule | Segment F1 with it | With no decoder |", "|---|---|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        with_own, with_none = (d["systems"][system]["unchanged_with"][key] for key in (OWN, NONE))
        setting = with_own["setting"]
        assert (setting["status"], setting["floors_dropped"], setting["nms_radius"], setting["threshold_quantile"]) == ("feasible", False, 5, 0.999)
        rows.append(f"| {system} | {plain(number(d, system))} | {f3(value(d, system, CLEAN, 'pairs'))} | {right(d, system)} | {wrong(d, system)} | "
                    f"half window {setting['half_window']}, threshold {f3(setting['threshold'])} | "
                    f"{f3(with_own['selection']['multi']['segment_f1_50']['value'])} | {f3(with_none['selection']['multi']['segment_f1_50']['value'])} |")
    return rows


def unchanged_difference_rows(d: dict) -> list[str]:
    rows = ["| On the unchanged frames | " + " | ".join(f"{a} minus {b}" for a, b in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    for title, _, part, key, digits, scale in NUMBERS:
        rows.append(f"| {title} | " + " | ".join(signed(apart(d, a, b, CLEAN, part, key), digits, scale) for a, b in PAIRS) + " |")
    return rows



def unchanged_text(d: dict) -> list[str]:
    def ruled(system: str) -> dict:
        return d["systems"][system]["unchanged_with"][OWN]

    def with_rule(system: str) -> float:
        return ruled(system)["selection"]["multi"]["segment_f1_50"]["value"]

    def with_none(system: str) -> float:
        return d["systems"][system]["unchanged_with"][NONE]["selection"]["multi"]["segment_f1_50"]["value"]

    twice = apart(d, PUBLISHED, AGAIN)
    assert abs(twice["value"]) < 0.005 and around_zero(twice)
    assert rounded(value(d, AGAIN), 4) == "0.9565" and rounded(with_rule(AGAIN), 4) == "0.9432" and ruled(AGAIN)["setting"]["half_window"] == 5
    order = sorted(SYSTEMS, key=lambda s: -value(d, s))
    assert order == [STAGE2, PUBLISHED, AGAIN, WITHOUT, MIRROR] and sorted(SYSTEMS, key=lambda s: -with_none(s)) == order
    among_four = [(a, b) for a, b in PAIRS if MIRROR not in (a, b)]
    assert len(among_four) == 5 and all(holds_zero(apart(d, a, b)) for a, b in among_four)
    assert below(apart(d, MIRROR, AGAIN)) and below(apart(d, MIRROR, PUBLISHED))
    # the other numbers of these five differences: which intervals lie on one side of zero
    sided = [(a, b, key) for a, b in among_four for _, _, part, key, _, _ in NUMBERS
             if above(apart(d, a, b, CLEAN, part, key)) or below(apart(d, a, b, CLEAN, part, key))]
    nearer = [(WITHOUT, STAGE2, "cut_distance_mean_s"), (AGAIN, STAGE2, "cut_distance_mean_s"), (PUBLISHED, STAGE2, "cut_distance_mean_s")]
    assert sorted(sided) == sorted([*nearer, (WITHOUT, STAGE2, "missed_segments")])
    assert all(below(apart(d, a, b, CLEAN, "multi", key)) for a, b, key in nearer)
    missed = apart(d, WITHOUT, STAGE2, CLEAN, "multi", "missed_segments")
    assert above(missed)
    pairs = {s: value(d, s, CLEAN, "pairs") for s in SYSTEMS}
    assert min(pairs, key=pairs.get) == STAGE2 and len({f3(pairs[s]) for s in SYSTEMS if s != STAGE2}) == 1
    assert all(holds_zero(apart(d, s, STAGE2, CLEAN, "pairs")) for s in (PUBLISHED, AGAIN, WITHOUT))
    windows = {s: ruled(s)["setting"]["half_window"] for s in SYSTEMS}
    assert windows == {PUBLISHED: 7, STAGE2: 7, AGAIN: 5, WITHOUT: 5, MIRROR: 7}
    threshold = {s: f3(ruled(s)["setting"]["threshold"]) for s in SYSTEMS}
    assert with_rule(MIRROR) > with_rule(AGAIN) > with_rule(WITHOUT) and value(d, AGAIN) > value(d, WITHOUT) > value(d, MIRROR)
    assert with_rule(AGAIN) < value(d, AGAIN) and with_rule(WITHOUT) < value(d, WITHOUT) and with_rule(MIRROR) > value(d, MIRROR)
    assert [right(d, s) for s in SYSTEMS] == [281, 281, 281, 280, 281] and [wrong(d, s) for s in SYSTEMS] == [1, 1, 1, 2, 1]

    def four(x: float) -> str:
        return rounded(x, 4)

    return [
        f"- **Trained again, the recipe comes out where it came out before.** The arm \"again\" has {f3(value(d, AGAIN))} where the published system has "
        f"{f3(value(d, PUBLISHED))}, and the published system minus \"again\" is {signed(twice)}. To four decimals the arm has 0.9565 with the published "
        "setting and 0.9432 with the setting that the rule gives it, a half window of 5. These are the two numbers of the rerun of 2026-11-20 "
        "([TRAINING.md](TRAINING.md)).",
        f"- **In segment F1 on the multi streams no arm is above the stage-2 ensemble.** With the published setting the stage-2 ensemble has "
        f"{f3(value(d, STAGE2))}, the published system {f3(value(d, PUBLISHED))}, \"again\" {f3(value(d, AGAIN))}, \"without input changes\" "
        f"{f3(value(d, WITHOUT))} and \"with a mirror\" {f3(value(d, MIRROR))}. The plan names five differences among the first four, and for segment F1 "
        f"the interval of each holds zero. The arm with a mirror is below \"again\" ({signed(apart(d, MIRROR, AGAIN))}) and below the published system "
        f"({signed(apart(d, MIRROR, PUBLISHED))}). With no decoder the five systems stand in the same order.",
        f"- **The other numbers of those five differences.** In the table above {numeral(len(sided))} of their intervals lie on one side of zero. The "
        "matched cuts of \"without input changes\", of \"again\" and of the published system lie nearer than those of the stage-2 ensemble: "
        f"{', '.join(signed(apart(d, a, b, CLEAN, 'multi', key)) for a, b, key in nearer)} seconds. And \"without input changes\" misses more segments "
        f"than the stage-2 ensemble ({signed(missed, 0)}). On the pair streams the stage-2 ensemble has the lowest value of the five, {f3(pairs[STAGE2])} "
        f"against {f3(pairs[PUBLISHED])} for each other system. Three of those four differences were scored, and their intervals hold zero.",
        f"- **Single recordings.** By the majority rule {right(d, WITHOUT)} of 284 are right for \"without input changes\" and {right(d, PUBLISHED)} for each "
        f"other system. `classify`'s rule reads {wrong(d, WITHOUT)} wrong for \"without input changes\" and {wrong(d, PUBLISHED)} for each other system.",
        "- **The rule's setting does not repeat, and with it the arm with a mirror goes from the lowest of the three arms to the highest.** On the "
        f"calibration half the rule gives the published system and the stage-2 ensemble a half window of 7 (thresholds {threshold[PUBLISHED]} and "
        f"{threshold[STAGE2]}), \"again\" and \"without input changes\" a half window of 5 ({threshold[AGAIN]} and {threshold[WITHOUT]}), and \"with a "
        f"mirror\" a half window of 7 ({threshold[MIRROR]}). To four decimals, with the rule's setting \"again\" has {four(with_rule(AGAIN))} where the "
        f"published setting gives it {four(value(d, AGAIN))}, \"without input changes\" has {four(with_rule(WITHOUT))} where it has "
        f"{four(value(d, WITHOUT))}, and \"with a mirror\" has {four(with_rule(MIRROR))} where it has {four(value(d, MIRROR))}. So with the rule's "
        "settings the arm with a mirror is above the two other arms, and with the published setting, which the plan fixed for every system, it is below "
        "both. No difference was scored with the rule's settings, so these differences have no interval.",
    ]


# ------------------------------------------------------------------------------------------------- the first question
def share_cell(found: dict) -> str:
    return "" if found["value"] is None else plain(found, DECIDES)


def family_rows(d: dict) -> list[str]:
    rows = ["| Condition | again | without input changes | stage 2 | \"again\" minus \"without input changes\" | \"again\" minus stage 2 | Share | "
            "Sentence | Share with the published system | Sentence | Published minus \"again\" | What stands |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in d["input_changes"]["conditions"].items():
        second = e["with_the_published_system"]
        rows.append(f"| {name} | {f3(value(d, AGAIN, name))} | {f3(value(d, WITHOUT, name))} | {f3(value(d, STAGE2, name))} | "
                    f"{signed(e['again_minus_without'], level=DECIDES)} | {signed(e['share']['of'], level=DECIDES)} | {share_cell(e['share'])} | "
                    f"{e['sentence']} | {share_cell(second['share'])} | {second['sentence']} | {signed(e['published_minus_again'], level=DECIDES)} | "
                    f"{e['stands'] if e['stands'] is not None else 'neither'} |")
    return rows


def family_text(d: dict) -> list[str]:
    family = d["input_changes"]["conditions"]
    assert d["input_changes"]["level"] == 99.0 and all(e["share"]["level"] == 99.0 and e["share"]["of"]["interval"][0] > 0 for e in family.values())
    for e in family.values():  # what the file says of a share is what its interval says
        for found, sentence in ((e["share"], e["sentence"]), (e["with_the_published_system"]["share"], e["with_the_published_system"]["sentence"])):
            low, high = found["interval"]
            assert sentence == (MOSTLY_CHANGES if low > 0.5 else NOT_TOLD) and high > 0.5
        agree = e["sentence"] == e["with_the_published_system"]["sentence"]
        assert e["the_two_trainings_agree"] is agree and e["stands"] == (e["sentence"] if agree else None)
        assert holds_zero(e["published_minus_again"])
    stands = {name: e["stands"] for name, e in family.items()}
    assert stands == {"noise 25": MOSTLY_CHANGES, "blur 2 pixels": MOSTLY_CHANGES, "64 pixels": NOT_TOLD, "speckle 0.25": None, "contrast 0.5": NOT_TOLD}
    speckle, contrast = family["speckle 0.25"], family["contrast 0.5"]
    assert (speckle["sentence"], speckle["with_the_published_system"]["sentence"]) == (NOT_TOLD, MOSTLY_CHANGES)
    assert [name for name, e in family.items() if not e["the_two_trainings_agree"]] == ["speckle 0.25"]
    assert min(family, key=lambda name: family[name]["share"]["of"]["value"]) == "contrast 0.5"  # the least to split
    no_share = {name: e["share"]["draws_without_a_share"]["streams"] for name, e in family.items()}
    no_share_published = {name: e["with_the_published_system"]["share"]["draws_without_a_share"]["streams"] for name, e in family.items()}
    assert {n: c for n, c in no_share.items() if c} == {"contrast 0.5": 10} and {n: c for n, c in no_share_published.items() if c} == {"contrast 0.5": 1}
    shares = [e["share"]["value"] for e in family.values()]
    assert min(shares) > 0.5
    loop = {name: apart(d, WITHOUT, STAGE2, name) for name in FAMILY}
    assert [name for name in FAMILY if above(loop[name])] == ["blur 2 pixels", "64 pixels"] and not any(below(loop[name]) for name in FAMILY)
    twice = [e["published_minus_again"]["value"] for e in family.values()]

    def share(name: str, published: bool = False) -> str:
        return plain(family[name]["with_the_published_system"]["share"] if published else family[name]["share"], DECIDES)

    return [
        f"- **Noise and blur: \"mostly the input changes\".** Under noise with a spread of 25 grey levels the share is {share('noise 25')}, and under a blur "
        f"of 2 pixels {share('blur 2 pixels')}. Both intervals lie wholly above one half, and the published system in the place of \"again\" gives the same "
        "sentence.",
        f"- **64 pixels: \"not told apart\".** The share is {share('64 pixels')}, and its interval holds one half. Both trainings say so.",
        f"- **Speckle: the two trainings give two sentences, and neither stands.** With \"again\" the share is {share('speckle 0.25')}, which is \"not told "
        f"apart\". With the published system it is {share('speckle 0.25', True)}, which is \"mostly the input changes\". The two lower ends lie on the two "
        "sides of one half.",
        f"- **Contrast: \"not told apart\".** The share is {share('contrast 0.5')}. What there is to split is the smallest of the five here, "
        f"{signed(contrast['share']['of'], level=DECIDES)} at 99%. In {no_share['contrast 0.5']} of the 10,000 draws it was not positive, and those draws "
        f"have no share; with the published system {no_share_published['contrast 0.5']} draw has none. Under the other four conditions every draw has a "
        "share.",
        f"- **As a number every share is above one half.** The five lie between {f3(min(shares))} and {f3(max(shares))}. The sentence follows the interval, "
        "and under three of the five conditions the interval does not decide.",
        f"- **The loop without input changes gains by itself under two of the five.** \"without input changes\" minus the stage-2 ensemble is "
        f"{signed(loop['blur 2 pixels'])} under the blur of 2 pixels and {signed(loop['64 pixels'])} at 64 pixels, at 95%. Under the other three the interval "
        "holds zero.",
        f"- **The published system in the place of \"again\".** Under the five conditions the published system minus \"again\" lies between "
        f"{rounded(min(twice), 3, True)} and {rounded(max(twice), 3, True)}, and each 99% interval holds zero. The two trainings give two sentences under "
        "speckle alone.",
    ]


# ------------------------------------------------------------------------------------------------ the second question
def mirror_rows(d: dict) -> list[str]:
    m = d["mirror"]
    gain, here, loss = m["under_the_mirrored_image"]["multi"]["segment_f1_50"], m["on_the_unchanged_frames"]["multi"]["segment_f1_50"], m["its_own_loss"]
    loss = loss["multi"]["segment_f1_50"]
    assert gain == apart(d, MIRROR, AGAIN, MIRRORED) and here == apart(d, MIRROR, AGAIN) and loss == own_loss(d, MIRROR, MIRRORED)
    assert m["margins"] == {"its_own_loss": 0.03, "on_the_unchanged_frames": 0.01}
    under = {name: apart(d, MIRROR, AGAIN, name) for name in FAMILY}
    fewer = right(d, AGAIN) - right(d, MIRROR)
    mine = {  # every condition of the plan, asked of the numbers again
        "gains under the mirrored image": above(gain),
        "stands the mirrored image": loss["value"] >= -0.03,
        "not below on the unchanged frames": not below(here) and here["value"] >= -0.01,
        "at most one recording fewer right": fewer <= 1,
        "not below under the five conditions": not any(below(under[name]) for name in FAMILY),
    }
    assert mine == m["holds"] and m["word"] == ("candidate" if all(mine.values()) else "not supported on development data")
    assert [name for name in FAMILY if below(under[name])] == ["noise 25"] and all(holds_zero(under[name]) for name in FAMILY if name != "noise 25")
    assert below(here) and here["value"] < -0.01  # both halves of the third condition fail

    def word(key: str) -> str:
        return "yes" if mine[key] else "no"

    return [
        "| What the plan asks of the arm with a mirror | What was found | Holds |", "|---|---|---|",
        f"| Under the mirrored image, the interval of \"with a mirror\" minus \"again\" lies wholly above zero | {signed(gain)} | "
        f"{word('gains under the mirrored image')} |",
        f"| Under the mirrored image, the arm loses no more than 0.03 against its own unchanged frames | {signed(loss)} | "
        f"{word('stands the mirrored image')} |",
        "| On the unchanged frames, the interval of \"with a mirror\" minus \"again\" does not lie wholly below zero, and the arm is no more than 0.01 "
        f"below | {signed(here)} | {word('not below on the unchanged frames')} |",
        f"| On the unchanged frames, at most one single recording fewer is read right by the majority rule | {right(d, MIRROR)} against {right(d, AGAIN)} | "
        f"{word('at most one recording fewer right')} |",
        f"| Under none of the five conditions of the first question the interval lies wholly below zero | noise 25: {signed(under['noise 25'])}; the other "
        f"four hold zero | {word('not below under the five conditions')} |",
    ]


def mirror_text(d: dict) -> list[str]:
    m, names = d["mirror"], conditions(d)
    assert m["word"] == "not supported on development data" and sum(m["holds"].values()) == 2
    losses = {s: own_loss(d, s, MIRRORED) for s in SYSTEMS}
    others = [-losses[s]["value"] for s in SYSTEMS if s != MIRROR]
    assert all(below(losses[s]) for s in SYSTEMS) and -losses[MIRROR]["value"] < min(others)
    here = d["differences"][f"{MIRROR} minus {AGAIN}"][CLEAN]
    assert here["recordings"]["accuracy"]["value"] == 0.0 and holds_zero(here["pairs"]["segment_f1_50"]) and right(d, MIRROR) == right(d, AGAIN)
    extra, missed = here["multi"]["extra_segments"], here["multi"]["missed_segments"]
    assert above(extra) and above(missed) and above(here["multi"]["wrong_samples"]) and below(here["multi"]["edit"])
    for key in ("cut_precision", "cut_recall", "cut_distance_mean_s", "same_view_cut_rate"):
        assert holds_zero(here["multi"][key])
    up, down = one_sided(d, MIRROR, AGAIN)
    assert up == [MIRRORED] and down[0] == CLEAN and len(down) == 12 and "noise 25" in down
    rest = [n for n in names if n not in (CLEAN, MIRRORED, *FAMILY)]  # the 20 conditions that the plan leaves to be described
    assert len(rest) == 20
    lower = [n for n in rest if apart(d, MIRROR, AGAIN, n)["value"] < 0]
    higher = [n for n in rest if apart(d, MIRROR, AGAIN, n)["value"] > 0]
    assert len(lower) + len(higher) == 20 and all(n in lower for n in down if n in rest)
    seeds = [seed_apart(d, MIRROR, k) for k in range(3)]
    assert all(below(s) for s in seeds)
    up_published, down_published = one_sided(d, MIRROR, PUBLISHED)
    assert up_published == [MIRRORED] and down_published[0] == CLEAN
    twice = {n: apart(d, PUBLISHED, AGAIN, n) for n in names}
    largest = max(abs(e["value"]) for e in twice.values())
    assert -here["multi"]["segment_f1_50"]["value"] > largest
    with_rule = {s: d["systems"][s]["unchanged_with"][OWN]["selection"]["multi"]["segment_f1_50"]["value"] for s in (AGAIN, MIRROR)}
    assert with_rule[MIRROR] > with_rule[AGAIN]
    return [
        f"- **The word is \"{m['word']}\".** Two of the five conditions hold and three do not.",
        f"- **What the mirror gives.** Under the mirrored image the arm with a mirror has {f3(value(d, MIRROR, MIRRORED))}, where \"again\" has "
        f"{f3(value(d, AGAIN, MIRRORED))} and the published system {f3(value(d, PUBLISHED, MIRRORED))}. By the majority rule {right(d, MIRROR, MIRRORED)} of "
        f"284 single recordings are right, against {right(d, AGAIN, MIRRORED)} and {right(d, PUBLISHED, MIRRORED)}, and `classify`'s rule reads "
        f"{wrong(d, MIRROR, MIRRORED)} wrong, against {wrong(d, AGAIN, MIRRORED)} and {wrong(d, PUBLISHED, MIRRORED)}.",
        f"- **It still reads a mirrored image worse than an unchanged one.** Against its own unchanged frames the arm loses {f3(-losses[MIRROR]['value'])} "
        f"under the mirrored image ({signed(losses[MIRROR])}), where the plan allowed 0.03. The other four systems lose {span(others)} there.",
        f"- **What it costs on the unchanged frames.** The arm with a mirror has {f3(value(d, MIRROR))} where \"again\" has {f3(value(d, AGAIN))}. The loss is "
        f"in the multi streams: {whole(extra['value'])} more segments that match nothing ({signed(extra, 0)}) and {whole(missed['value'])} more segments "
        f"missed ({signed(missed, 0)}). On the pair streams the interval holds zero ({signed(here['pairs']['segment_f1_50'])}), and of the single "
        f"recordings {right(d, MIRROR)} are right for both arms.",
        f"- **The cost is not only on the unchanged frames.** The interval of \"with a mirror\" minus \"again\" lies wholly below zero under {len(down)} of "
        f"the 27 conditions: the unchanged frames, {listed(down[1:])}. It lies above zero under the mirrored image alone. Under the 20 conditions that the "
        f"plan leaves to be described, the arm with a mirror has the lower value under {len(lower)} and the higher under {len(higher)} "
        f"({listed(higher)}), and the interval lies below zero under {len([n for n in down if n in rest])}.",
        f"- **With the settings of the rule the order on the unchanged frames turns.** The arm with a mirror then has {f3(with_rule[MIRROR])} and \"again\" "
        f"{f3(with_rule[AGAIN])} (the table of the unchanged frames). The plan fixed the published setting for the word, and the difference with the rule's "
        "settings has no interval.",
        f"- **Every seed shows the loss.** Seed for seed the arm with a mirror is below \"again\" on the unchanged frames: {signed(seeds[0])}, "
        f"{signed(seeds[1])} and {signed(seeds[2])}.",
        f"- **Against the published system** the arm with a mirror is {signed(apart(d, MIRROR, PUBLISHED, MIRRORED))} under the mirrored image and "
        f"{signed(apart(d, MIRROR, PUBLISHED))} on the unchanged frames, and the interval lies below zero under {len(down_published)} of the 27 conditions.",
        f"- **Beside two trainings of one recipe.** Over the 27 conditions the published system minus \"again\" lies between "
        f"{rounded(min(e['value'] for e in twice.values()), 3, True)} and {rounded(max(e['value'] for e in twice.values()), 3, True)}. The loss of the arm "
        f"with a mirror on the unchanged frames, {f3(-here['multi']['segment_f1_50']['value'])}, is larger than the largest of these differences, "
        f"{f3(largest)}. One run for every seed and arm cannot say more than that.",
    ]


# ------------------------------------------------------------------------------------------------- under every condition
def value_rows(d: dict) -> list[str]:
    rows = ["| Segment F1 at 50%, multi streams | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    return rows + [f"| {name} | " + " | ".join(f3(value(d, s, name)) for s in SYSTEMS) + " |" for name in conditions(d)]


def loss_rows(d: dict) -> list[str]:
    rows = ["| Minus its own unchanged frames | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    return rows + [f"| {name} | " + " | ".join(signed(own_loss(d, s, name)) for s in SYSTEMS) + " |" for name in conditions(d) if name != CLEAN]


def recording_rows(d: dict) -> list[str]:
    rows = ["| Right of 284 by the majority rule, and wrong by `classify`'s rule | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    return rows + [f"| {name} | " + " | ".join(f"{right(d, s, name)}, {wrong(d, s, name)}" for s in SYSTEMS) + " |" for name in conditions(d)]


def difference_rows(d: dict) -> list[str]:
    rows = ["| Segment F1 at 50%, multi streams | " + " | ".join(f"{a} minus {b}" for a, b in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    return rows + [f"| {name} | " + " | ".join(signed(apart(d, a, b, name)) for a, b in PAIRS) + " |" for name in conditions(d)]


def every_number_rows(d: dict, first: str, second: str) -> list[str]:
    """One row for every condition and one column for every number of section 5, for one difference."""
    for name in conditions(d):  # a count of confused recordings has an interval exactly where both counts are above 5
        for key in ("a4c_as_a5c", "a5c_as_a4c"):
            counts = [value(d, system, name, "recordings", key) for system in (first, second)]
            assert (apart(d, first, second, name, "recordings", key)["interval"] is not None) == (min(counts) > 5)
    rows = ["| Condition | " + " | ".join(short for _, short, _, _, _, _ in NUMBERS) + " |", "|---|" + "---|" * len(NUMBERS)]
    for name in conditions(d):
        cells = [signed(apart(d, first, second, name, part, key), digits, scale) for _, _, part, key, digits, scale in NUMBERS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return rows


def conditions_text(d: dict, part2: dict) -> list[str]:
    names = conditions(d)
    changes_up, changes_down = one_sided(d, AGAIN, WITHOUT)
    assert changes_down == [] and CLEAN not in changes_up and len(changes_up) == 14
    largest = sorted(names, key=lambda n: -apart(d, AGAIN, WITHOUT, n)["value"])[:4]
    assert largest == ["noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25"] and all(n in changes_up for n in largest)
    loop_up, loop_down = one_sided(d, WITHOUT, STAGE2)
    assert (loop_up, loop_down) == (["blur 2 pixels", "64 pixels"], ["zoomed out 0.8"])
    loop_here = apart(d, WITHOUT, STAGE2)
    assert around_zero(loop_here) and len(names) - len(loop_up) - len(loop_down) == 24  # neither a loss nor a gain is excluded on the unchanged frames
    stage_up, stage_down = one_sided(d, AGAIN, STAGE2)
    assert stage_down == [] and len(stage_up) == 11 and all(n in stage_up for n in FAMILY)
    published_up, published_down = one_sided(d, PUBLISHED, STAGE2)
    assert published_down == [] and all(n in published_up for n in FAMILY)
    by_size = sorted(names, key=lambda n: -apart(d, PUBLISHED, STAGE2, n)["value"])
    assert by_size[:5] == list(FAMILY)
    before = {n: part2["published_minus_stage2"][n]["multi"]["segment_f1_50"]["value"] for n in FAMILY}
    now = {n: apart(d, PUBLISHED, STAGE2, n)["value"] for n in FAMILY}
    assert all(now[n] < before[n] for n in FAMILY)
    assert f3(part2["systems"][STAGE2]["setting"]["params"]["threshold"]) == "0.185"
    twice_up, twice_down = one_sided(d, PUBLISHED, AGAIN)
    assert twice_up == [] and twice_down == ["rotated 10 degrees", "half the frame rate"]
    # an end that prints as zero and decides on which side an interval is counted
    ends = {pair: {n: apart(d, *pair, n)["interval"] for n in names} for pair in PAIRS}
    just_above = {pair: [n for n in names if 0 < ends[pair][n][0] < 0.0005] for pair in PAIRS}
    at_zero = {pair: [n for n in names if ends[pair][n][0] == 0] for pair in PAIRS}
    both_zero = {pair: [n for n in names if ends[pair][n][0] == 0 == ends[pair][n][1]] for pair in PAIRS}
    just_below = {pair: [n for n in names if -0.0005 < ends[pair][n][1] < 0] for pair in PAIRS}
    assert not any(-0.0005 < ends[pair][n][0] < 0 for pair in PAIRS for n in names)  # no lower end prints as -0.000: the bullet has no sentence for one
    assert all(any(found.values()) for found in (just_above, at_zero, both_zero, just_below))
    for pair in PAIRS:
        for n in names:
            low, high = ends[pair][n]
            assert not 0 <= high < 0.0005 or low <= 0  # an upper end printed as +0.000: the interval holds zero
            assert n not in just_above[pair] or high >= 0.0005  # where the lower end is just above zero, the upper end prints as a number

    def where(found: dict) -> str:
        return "; ".join(f"{NAMED[a]} minus {NAMED[b]} under {listed(found[(a, b)])}" for a, b in PAIRS if found[(a, b)])

    return [
        f"- **\"again\" minus \"without input changes\".** The interval lies above zero under {len(changes_up)} of the 26 changed conditions and below zero "
        f"under none. The four largest differences are under noise 25, blur 2 pixels, 64 pixels and speckle 0.25: "
        f"{', '.join(rounded(apart(d, AGAIN, WITHOUT, n)['value'], 3, True) for n in largest)}. On the unchanged frames the two arms are not told apart "
        f"({signed(apart(d, AGAIN, WITHOUT))}).",
        f"- **\"without input changes\" minus the stage-2 ensemble.** The interval lies above zero under the blur of 2 pixels and at 64 pixels, and below zero "
        f"when the image is zoomed out to 0.8 ({signed(apart(d, WITHOUT, STAGE2, 'zoomed out 0.8'))}). Under the other 24 conditions it holds zero, the "
        f"unchanged frames among them ({signed(loop_here)}). Section 12 said that this arm sees no gamma of 0.9 and that a loss on the unchanged streams "
        "could come from there. Such a loss is neither shown nor excluded.",
        f"- **\"again\" minus the stage-2 ensemble.** The interval lies above zero under {len(stage_up)} conditions, the five of the first question among "
        "them, and below zero under none.",
        f"- **The published system minus the stage-2 ensemble, both with the published setting.** The interval lies above zero under {len(published_up)} "
        "conditions and below zero under none. The five conditions of the first question are still the five largest differences: "
        f"{', '.join(rounded(now[n], 3, True) for n in FAMILY)}. Each is smaller than in part 2 "
        f"({', '.join(rounded(before[n], 3, True) for n in FAMILY)}), where the stage-2 ensemble had the rule's threshold of 0.185.",
        f"- **Two trainings of one recipe.** The interval of the published system minus \"again\" lies above zero under no condition and below zero under two: "
        f"rotated 10 degrees ({signed(apart(d, PUBLISHED, AGAIN, 'rotated 10 degrees'))}) and half the frame rate "
        f"({signed(apart(d, PUBLISHED, AGAIN, 'half the frame rate'))}).",
        "- **Ends that print as zero.** The counts above follow the file and not the printed digits. A lower end printed as +0.000 is above zero, and the "
        f"interval counts as lying above zero, for {where(just_above)}. It is exactly zero, and the interval holds zero, for {where(at_zero)}. Both ends "
        f"are zero for {where(both_zero)}. An upper end printed as -0.000 is below zero, and the interval counts as lying below zero, for "
        f"{where(just_below)}. An upper end printed as +0.000 is zero or above it, and the interval holds zero.",
    ]


# ---------------------------------------------------------------------------------------------------------- the seeds
def seed_rows(d: dict) -> list[str]:
    given = (PUBLISHED, *ARMS)
    rows = ["| Segment F1 at 50%, multi streams, unchanged frames | seed 0 | seed 1 | seed 2 | The three together |", "|---|---|---|---|---|"]
    for system in given:
        alone = [d["seeds"]["systems"][f"{system} | seed {k}"]["multi"]["segment_f1_50"]["value"] for k in range(3)]
        rows.append(f"| {system} | " + " | ".join(f3(x) for x in alone) + f" | {f3(value(d, system))} |")
    rows += ["", "| The seed minus the seed of \"again\" of the same number | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
    for system in (PUBLISHED, WITHOUT, MIRROR):
        cells = [signed(seed_apart(d, system, k)) for k in range(3)]
        rows.append(f"| {system} | " + " | ".join(cells) + " |")
    return rows


def seed_text(d: dict) -> list[str]:
    def pair(system: str, k: int) -> dict:
        return seed_apart(d, system, k)

    assert all(below(pair(MIRROR, k)) for k in range(3)) and all(holds_zero(pair(WITHOUT, k)) for k in range(3))
    assert holds_zero(pair(PUBLISHED, 0)) and below(pair(PUBLISHED, 1)) and above(pair(PUBLISHED, 2))
    assert abs(pair(PUBLISHED, 0)["value"]) < 0.004 and abs(pair(PUBLISHED, 1)["value"]) < 0.004 < abs(pair(PUBLISHED, 2)["value"])
    return [
        "- **With a mirror:** each of the three seeds is below its partner, and each interval lies below zero.",
        "- **Without input changes:** each of the three intervals holds zero.",
        f"- **The published seeds against the seeds trained again:** seed 1 is below its partner ({signed(pair(PUBLISHED, 1))}), seed 2 is above "
        f"({signed(pair(PUBLISHED, 2))}), and for seed 0 the interval holds zero. Two trainings of one recipe differ seed by seed, in both directions.",
        "- **Beside the 0.004 of section 12.** Section 12 cites seeds trained again within 0.004 of the published ones. Those were the seeds of the "
        "rerun of 2026-11-20, each with the setting that the rule gives it ([TRAINING.md](TRAINING.md)). The seeds of the arm \"again\" were "
        "trained for part 4, and here they and the published seeds have the published setting: seeds 0 and 1 lie within 0.004 and seed 2 lies "
        f"{f3(abs(pair(PUBLISHED, 2)['value']))} apart.",
    ]


# ------------------------------------------------------------------------------------------- against what was expected
def expected_text(d: dict) -> list[str]:
    family = d["input_changes"]["conditions"]
    twice = apart(d, PUBLISHED, AGAIN)["value"]
    assert abs(twice) < 0.005
    stands = {name: e["stands"] for name, e in family.items()}
    assert [name for name in FAMILY if stands[name] == MOSTLY_CHANGES] == ["noise 25", "blur 2 pixels"]
    assert stands["64 pixels"] == NOT_TOLD and stands["speckle 0.25"] is None and stands["contrast 0.5"] == NOT_TOLD
    loop = apart(d, WITHOUT, STAGE2)["value"]
    about = {arm: [e["from_its_start"] for e in d["systems"][arm]["system"]["distances"]] for arm in ARMS}
    assert abs(loop) < 0.01 and all(about[WITHOUT][k] < about[AGAIN][k] for k in range(3))
    loss, here = -own_loss(d, MIRROR, MIRRORED)["value"], -apart(d, MIRROR, AGAIN)["value"]
    assert d["mirror"]["word"] == "not supported on development data" and loss > 0.03 and here > 0.005
    return [
        f"- **Trained again: as expected.** The arm \"again\" lies {f3(abs(twice))} from the published system on the unchanged frames, where 0.005 at most "
        "was expected.",
        "- **The first question: right for three conditions, wrong for two.** \"mostly the input changes\" was expected for the noise, the blur, the 64 "
        "pixels and the speckle, and \"not told apart\" for the contrast. The noise, the blur and the contrast came out so. At 64 pixels the sentence is "
        "\"not told apart\", and for the speckle the two trainings give two sentences and neither stands.",
        f"- **Without input changes: as expected.** On the unchanged frames the arm lies {f3(abs(loop))} from the stage-2 ensemble, where 0.01 at most was "
        "expected, and for every seed its weights lie nearer to their start than those of \"again\".",
        f"- **The second question: wrong.** The word \"candidate\" was expected, and the word is \"not supported on development data\". Under the mirrored "
        f"image the arm loses {f3(loss)} against its own unchanged frames, where less than 0.03 was expected. On the unchanged frames it lies {f3(here)} "
        "below \"again\", where 0.005 at most was expected.",
    ]


# -------------------------------------------------------------------------------------------------- the whole section
def part_four(d: dict, part2: dict, part2_sha256: str) -> str:
    """Section 13: stage 3 trained again, three ways. `part2` is the output of part 2, to which part 4 is held, and
    `part2_sha256` the checksum of its file."""
    if not __debug__:
        raise SystemExit("the checks of this tool are assertions: run it without -O")
    consistent(d, part2)
    held_to_the_package(d)
    worse = one_sided(d, MIRROR, AGAIN)[1]  # the conditions under which the arm with a mirror is below "again", the unchanged frames among them
    assert worse[0] == CLEAN and above(apart(d, MIRROR, AGAIN, MIRRORED))
    names = conditions(d)
    as_in_part2 = [n for n in names if value(d, STAGE2, n) == part2["systems"][STAGE2]["conditions"][n]["selection"]["multi"]["segment_f1_50"]["value"]]
    assert CLEAN not in as_in_part2  # the stage-2 ensemble has another decoder setting here than in part 2
    lines = [
        "## 13. Results of part 4",
        "",
        *run_text(d, part2, part2_sha256),
        "",
        "**The training itself.** The loss is the one the loop logs, with 0.15 times its penalty. A distance between two sets of weights is given as a share "
        "of the length of the second.",
        "",
        *model_rows(d),
        "",
        *training_text(d),
        "",
        "**On the unchanged frames.** Segment F1, the pair streams and the recordings by the majority rule are with the published setting, and "
        "`classify`'s rule uses no decoder. The last three columns give the setting that the rule chooses for a system on the calibration half, segment F1 "
        "on the multi streams with it, and segment F1 with no decoder.",
        "",
        *unchanged_rows(d),
        "",
        "Every number of section 5 on the unchanged frames, for the seven differences, with the same draws. A count of confused recordings has an interval "
        "only where both counts are above 5.",
        "",
        *unchanged_difference_rows(d),
        "",
        *unchanged_text(d),
        "",
        "**The first question: what the input changes add.** Segment F1 at 50% on the multi streams under the five conditions. The share is \"again\" minus "
        "\"without input changes\", divided by \"again\" minus the stage-2 ensemble. Every interval of this table is at 99%, and so is every share "
        "in the lines below it.",
        "",
        *family_rows(d),
        "",
        *family_text(d),
        "",
        "**The second question: the arm with a mirror.** Segment F1 at 50% on the multi streams unless a line names another number.",
        "",
        *mirror_rows(d),
        "",
        *mirror_text(d),
        "",
        "**Every system under every condition.** Segment F1 at 50% on the multi streams. The numbers of the published system are those of part 2. Those of the "
        f"stage-2 ensemble are not: here it has the published setting, and in the files its value differs from that of part 2 under "
        f"{len(names) - len(as_in_part2)} of the 27 conditions.",
        "",
        *value_rows(d),
        "",
        "Each system under a condition minus the same system on the unchanged frames:",
        "",
        *loss_rows(d),
        "",
        "Single recordings. Each cell gives how many of the 284 are right by the majority rule and, after the comma, how many `classify`'s rule reads wrong:",
        "",
        *recording_rows(d),
        "",
        "**The seven differences under every condition,** with the same draws.",
        "",
        *difference_rows(d),
        "",
        *conditions_text(d, part2),
        "",
        "**The seeds.** Every seed alone on the unchanged frames, decoded with the published setting. In part 1 every seed had the setting that the rule "
        "gives it.",
        "",
        *seed_rows(d),
        "",
        *seed_text(d),
        "",
        "**Against what was expected** (section 12).",
        "",
        *expected_text(d),
        "",
        "**What follows.** By section 6 nothing changes in the published system, and the mirrored image stays a limit of it. The arm with a mirror is not a "
        "candidate. What part 4 shows of it is a trade, on development data and under changes that this experiment made up. A model that was fed mirrored "
        f"frames has {f3(value(d, MIRROR, MIRRORED))} under the mirrored image where the same recipe without them has {f3(value(d, AGAIN, MIRRORED))}. With "
        f"this recipe the interval of its difference to \"again\" lies below zero on the unchanged frames and under {numeral(len(worse) - 1)} other "
        "conditions. Only one probability of the mirror and one length of training were tried. Whether another choice lowers the price was not measured, and "
        "a plan for it would be written knowing these results.",
        "",
        "For the first question, the input changes are most of what stage 3 adds under the noise with a spread of 25 grey levels and under the blur of 2 "
        "pixels. At 64 pixels, under speckle and at a contrast of 0.5 part 4 does not tell them from the loop. These are made-up changes of development "
        "recordings. The agreement on MIMIC recordings was not scored, so part 4 does not say what brought the gain of stage 3 there.",
        "",
        "The README and the model card said of the mirrored image: \"No training stage mirrored a frame; whether that is the cause was not tested.\" Part 4 "
        "trained models with mirrored frames, so the words \"was not tested\" no longer hold. Section 12 allowed the two documents one added sentence, that "
        "a model trained with a mirror stands a mirrored image on development data, and only for a candidate. They do not get that sentence. In the place of "
        "the words that no longer hold they give the gain and the cost in numbers, for the arm with a mirror and for the arm \"again\", and say that by "
        "the rule of section 12 these models are no candidate and are not published. This goes beyond what section 12 named, and it is said here for that "
        "reason.",
        "",
        "The two documents also said that the input changes and the further updates of stage 3 were not told apart. Section 12 said that this sentence "
        "stays whatever part 4 shows, and it read the sentence as one about the agreement on MIMIC recordings. Part 4 did not score that agreement, and "
        "under made-up changes it now has runs that tell the two apart. So the sentence was given the words \"for this agreement\". Its meaning is the "
        "one section 12 gave it, and its wording is changed where section 12 said that it stays.",
        "",
        "**Every number of section 5 for the seven differences.** The tables above show a part. A count of confused recordings has an interval only where "
        "both counts are above 5. The file also holds every number of section 5 for every system under every condition, for its difference to the "
        "unchanged frames, for the settings of the rule and for the seeds.",
    ]
    for first, second in PAIRS:
        lines += ["", f"{first} minus {second}:", "", *every_number_rows(d, first, second)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise SystemExit(__doc__)
    result, part2 = (Path(name) for name in args)
    sys.stdout.write(part_four(json.loads(result.read_text()), json.loads(part2.read_text()), hashlib.sha256(part2.read_bytes()).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
