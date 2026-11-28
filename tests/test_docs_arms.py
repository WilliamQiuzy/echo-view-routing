"""Part 4 of docs/ABLATIONS.md is held to its output file.

The results of section 13 are written by tools/write_arms_results.py from docs/ablations/arms.json and from the output
of part 2. The document must hold what the tool prints, to the letter, and a file that says something else cannot be
written up. The second test reaches the headline numbers without the tool, holds the plan to the text that was
committed before the run, and holds the sentences of the other documents. Nothing here runs a model.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ABLATIONS = "docs/ABLATIONS.md"
SINGLE = 284  # the single recordings of the selection half
PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR = "published", "stage 2", "again", "without input changes", "with a mirror"
ARMS = (AGAIN, WITHOUT, MIRROR)
FAMILY = ("noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5")
MOSTLY, NOT_TOLD = "mostly the input changes", "not told apart"
NOT_SUPPORTED = "not supported on development data"
NUMERAL = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
# Section 12, the plan of part 4, as commit 07bf68e holds it: the text that was committed before any model was trained.
PLAN_SHA256 = "8afa935b4c1d19e7bd29c3b4754120e550a9cc0e66fff93a36fc1feead6bf9a7"
STATUS_THEN = "**Status: planned on 2026-11-26 and committed on 2026-11-27 (UTC), before any of its models was trained. Not run yet.**"
STATUS_NOW = ("**Status: planned on 2026-11-26 and committed on 2026-11-27 (UTC), before any of its models was trained. Run on 2026-11-27 and 2026-11-28; "
              "the results are in section 13.**")


def read(path: str) -> str:
    return (ROOT / path).read_text()


def stored(name: str) -> dict:
    return json.loads((DOCS / "ablations" / name).read_text())


def said(document: str, *phrases: str) -> None:
    """Every phrase stands in the document, word for word."""
    text = read(document)
    for phrase in phrases:
        assert phrase in text, f"{document} does not say: {phrase}"


def shown(x: float, digits: int = 3, sign: bool = False) -> str:
    """A number as the documents show it: it is taken at twelve decimals, so that a tie which the file stores a hair
    off itself is the tie, and a tie of the last digit goes away from zero. Written here a second time, so that the
    tool's own rounding is not what holds it."""
    value = Decimal(repr(float(x))).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def f3(x: float) -> str:
    return shown(x)


def spread(entry: dict) -> str:
    low, high = entry["interval"]
    return f"{shown(entry['value'], 3, True)} [{shown(low, 3, True)}, {shown(high, 3, True)}]"


def writer_of_the_results():
    spec = importlib.util.spec_from_file_location("write_arms_results", ROOT / "tools" / "write_arms_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checksum_of_part_two() -> str:
    return hashlib.sha256((DOCS / "ablations" / "forward.json").read_bytes()).hexdigest()


def test_part_four_of_the_ablations_is_what_the_tool_writes_from_its_file(capsys):
    writer = writer_of_the_results()
    arms, forward, digest = stored("arms.json"), stored("forward.json"), checksum_of_part_two()
    section = writer.part_four(arms, forward, digest)
    document = read(ABLATIONS)
    assert section.startswith("## 13. Results of part 4\n\n**Run from 2026-11-27 01:34 UTC to 2026-11-28 04:58 UTC with the code of commit `07bf68e`")
    assert document.endswith(section) and document.count("## 13. Results of part 4") == 1  # the section is the end of the document, and it stands once
    # the command prints the same, and it takes the two files and nothing else
    assert writer.main([str(DOCS / "ablations" / "arms.json"), str(DOCS / "ablations" / "forward.json")]) == 0
    assert capsys.readouterr().out == section
    with pytest.raises(SystemExit):
        writer.main([str(DOCS / "ablations" / "arms.json")])

    # a file that says something else cannot be written up: a check of the tool stops it, and it is the check that the change is named for
    def refused(change, by: str, checksum: str = digest) -> None:
        other, part2 = copy.deepcopy(arms), copy.deepcopy(forward)
        change(other, part2)
        with pytest.raises(AssertionError) as caught:
            writer.part_four(other, part2, checksum)
        assert caught.traceback[-1].name == by, f"{change.__name__} was refused by {caught.traceback[-1].name}"

    def unchanged(first: str, second: str, other: dict) -> dict:
        return other["differences"][f"{first} minus {second}"]["unchanged"]["multi"]["segment_f1_50"]

    def a_candidate(other, part2):
        other["mirror"]["word"] = "candidate"

    def a_run_that_went_on(other, part2):
        other["systems"][MIRROR]["system"]["models"][1]["sessions"] = 2

    def a_sentence_that_stands_alone(other, part2):
        other["input_changes"]["conditions"]["speckle 0.25"]["stands"] = MOSTLY

    def an_interval_that_holds_zero(other, part2):  # "with a mirror" minus "again" on the unchanged frames
        unchanged(MIRROR, AGAIN, other)["interval"] = [-0.027, 0.003]

    def another_part_two(other, part2):
        part2["systems"][PUBLISHED]["conditions"]["noise 25"]["selection"]["multi"]["segment_f1_50"]["value"] += 1e-9

    def another_commit(other, part2):
        other["code"] = "abc1234"

    def logits_that_do_not_agree(other, part2):
        other["unchanged_against_own_logits"][WITHOUT] = 0.0004

    def a_seed_that_shows_no_loss(other, part2):
        other["seeds"]["minus_again"][MIRROR]["seed 1"]["multi"]["segment_f1_50"]["interval"] = [-0.019, 0.001]

    def a_difference_that_is_not_one(other, part2):  # no longer the first value minus the second
        unchanged(PUBLISHED, AGAIN, other)["value"] = 0.0061

    def a_number_more(other, part2):
        other["differences"][f"{AGAIN} minus {WITHOUT}"]["noise 10"]["multi"]["another number"] = {"value": 0.0, "level": 95.0, "interval": None}

    def a_difference_with_another_setting(other, part2):
        by = other["differences"][f"{MIRROR} minus {AGAIN}"]
        by["the setting by the rule"] = copy.deepcopy(by["unchanged"])

    def a_published_weight_file(other, part2):
        from echoview import weights

        digest_of_one = next(iter(weights.START_SHA256.values()))
        for arm in ARMS:
            about = other["systems"][arm]["system"]
            if arm == AGAIN:
                about["weights_sha256"][0] = about["models"][0]["sha256"] = digest_of_one
            else:
                about["set_against_sha256"][0] = digest_of_one

    def another_option(other, part2):
        other["systems"][WITHOUT]["system"]["models"][2]["identity"]["lr_new"] = 0.0002

    def another_level(other, part2):
        unchanged(MIRROR, AGAIN, other)["level"] = 99.0

    def another_level_in_the_first_question(other, part2):
        other["input_changes"]["conditions"]["noise 25"]["published_minus_again"]["level"] = 95.0

    def an_interval_that_ends_at_zero(other, part2):  # "without input changes" minus stage 2 on the unchanged frames: a gain would be excluded
        unchanged(WITHOUT, STAGE2, other)["interval"][1] = 0.0

    def a_value_outside_its_interval(other, part2):
        other["input_changes"]["conditions"]["64 pixels"]["again_minus_without"]["interval"] = [0.2, 0.3]

    def a_share_that_is_not_one(other, part2):
        other["input_changes"]["conditions"]["noise 25"]["share"]["value"] = 0.95

    def a_number_less_for_a_seed(other, part2):
        del other["seeds"]["systems"][f"{AGAIN} | seed 1"]["multi"]["edit"]

    def another_version_in_part_two(other, part2):
        part2["systems"][STAGE2]["system"]["torch"] = "2.13.0"

    def an_interval_that_prints_as_zero_twice(other, part2):  # "again" minus "without input changes": wholly above zero, and both ends print as +0.000
        entry = other["differences"][f"{AGAIN} minus {WITHOUT}"]["rotated 10 degrees"]["multi"]["segment_f1_50"]
        entry["interval"] = [0.0001, 0.0003]

    def nothing(other, part2):
        pass

    for change, by in ((a_candidate, "mirror_rows"), (a_run_that_went_on, "training_text"), (a_sentence_that_stands_alone, "family_text"),
                       (an_interval_that_holds_zero, "part_four"), (another_part_two, "run_text"), (another_commit, "run_text"),
                       (logits_that_do_not_agree, "run_text"), (a_seed_that_shows_no_loss, "mirror_text"), (a_difference_that_is_not_one, "consistent"),
                       (a_number_more, "consistent"), (a_difference_with_another_setting, "consistent"), (a_published_weight_file, "held_to_the_package"),
                       (another_option, "held_to_the_package"), (another_level, "at"), (another_level_in_the_first_question, "at"),
                       (an_interval_that_ends_at_zero, "conditions_text"), (a_value_outside_its_interval, "at"), (a_share_that_is_not_one, "consistent"),
                       (a_number_less_for_a_seed, "consistent"), (another_version_in_part_two, "consistent"),
                       (an_interval_that_prints_as_zero_twice, "at")):
        refused(change, by)
    refused(nothing, "run_text", checksum="0" * 64)  # another file of part 2 than the one the report was held to

    # a tie of the last digit goes away from zero, also where the file stores it a hair below the tie
    assert writer.rounded(0.9594999999999999, 3) == "0.960" == shown(0.9594999999999999) and writer.rounded(15352 / 16000, 3) == "0.960"
    assert writer.rounded(-0.0024999999999999467, 3, True) == "-0.003" and writer.rounded(0.0045, 1, False, 100.0) == "0.5"
    assert writer.rounded(0.12345, 3) == "0.123" and writer.rounded(-0.00013, 3, True) == "-0.000" and writer.rounded(0.0, 3, True) == "+0.000"


def test_the_tools_of_the_results_do_not_run_without_their_checks():
    """The checks of the two tools are assertions, and `python -O` drops assertions: then nothing is printed."""
    files = [str(DOCS / "ablations" / name) for name in ("arms.json", "forward.json")]
    for command in ([str(ROOT / "tools" / "write_arms_results.py"), *files], [str(ROOT / "tools" / "write_ablation_results.py"), "part2", files[1]]):
        done = subprocess.run([sys.executable, "-O", *command], capture_output=True, text=True, check=False)
        assert done.returncode != 0 and done.stdout == "" and "without -O" in done.stderr


def test_the_code_that_runs_a_condition_is_the_same_in_the_commits_of_parts_two_and_four():
    """Section 13 scores the files that part 2 wrote with commit fb674f6 beside the arms that commit 07bf68e ran. Where
    the history of the repository is at hand, the files on the path of a forward pass are the same in both commits."""
    files = [f"src/echoview/{name}" for name in ("pipeline.py", "model.py", "frames.py", "windows.py", "weights.py", "reference.py", "decoder.py")]
    files += [f"src/echoview/evaluation/{name}" for name in ("stress.py", "ev9v.py", "run.py", "benchmark.py", "comparison.py")]
    for commit in ("fb674f6", "07bf68e"):
        if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit + "^{commit}"], capture_output=True, check=False).returncode != 0:
            pytest.skip("the two commits are not in this checkout")
    assert all((ROOT / name).is_file() for name in files)
    apart = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "fb674f6", "07bf68e", "--", *files], capture_output=True, text=True, check=True)
    assert apart.stdout.strip() == ""
    said(ABLATIONS, "with the code of commit `fb674f6`, which is the plan.**", "The code that runs a system under a condition is the same in both commits.")


def test_the_headline_numbers_of_part_four_stand_in_the_documents():
    from echoview import weights
    from echoview.evaluation import arms as code

    arms, forward, part_one = stored("arms.json"), stored("forward.json"), stored("stored.json")
    raw = (DOCS / "ablations" / "arms.json").read_text()
    # the run: one commit, the draws of the other parts, the files of parts 1 and 2 under it, pooled numbers only
    run = (arms["benchmark"], arms["half"], arms["sequences"], arms["draws"], arms["seed"], arms["code"])
    assert run == ("development", "selection", 541, 10_000, 20261123, "07bf68e")
    assert arms["sha256"]["part2"] == checksum_of_part_two()
    assert arms["sha256"]["logits"] == forward["sha256"]["logits"] == part_one["sha256"]["logits"]
    assert not re.search(r"(?<![0-9a-f.])(?=\d*[a-f])[0-9a-f]{16}(?![0-9a-f])", raw) and not re.search(r"NaN|Infinity", raw)
    assert tuple(arms["systems"]) == code.SYSTEMS == (PUBLISHED, STAGE2, AGAIN, WITHOUT, MIRROR)
    assert tuple(arms["differences"]) == tuple(f"{first} minus {second}" for first, second in code.DIFFERENCES)
    assert arms["decoder"]["half_window"] == 7 and f3(arms["decoder"]["threshold"]) == "0.174"

    # the plan the results are judged by is the text that was committed before the run: only its line of status has changed
    document = read(ABLATIONS)
    plan = document[document.index("## 12. Part 4: stage 3 trained again, three ways"):document.index("## 13. Results of part 4")]
    assert plan.count(STATUS_NOW) == 1 and hashlib.sha256(plan.replace(STATUS_NOW, STATUS_THEN).encode()).hexdigest() == PLAN_SHA256
    for expected in ("- **Trained again.** On the unchanged frames the arm \"again\" lies within 0.005 of the published system in segment F1 on the multi "
                     "streams.",
                     "- **The first question.** The sentence is \"mostly the input changes\" for the noise, the blur, the 64 pixels and the speckle, and "
                     "\"not told apart\" for the contrast.",
                     "- **Without input changes.** On the unchanged frames this arm lies within 0.01 of the stage-2 ensemble at the published setting, and "
                     "its weights lie nearer to their start than those of the arm \"again\".",
                     "- **The second question.** The word is \"candidate\". Under the mirrored image the arm with a mirror loses less than 0.03 against its "
                     "own unchanged frames, and on the unchanged frames it lies within 0.005 of the arm \"again\"."):
        assert expected in plan
    said(ABLATIONS, "it was planned after the results of parts 1 to 3 were read (section 12) and run on 2026-11-27 and 2026-11-28 (section 13).**")
    judged = document[document.index("**Against what was expected** (section 12)."):]
    margins = ("where 0.005 at most was expected", "where 0.01 at most was expected", "where less than 0.03 was expected")
    assert [judged.count(words) for words in margins] == [2, 1, 1]

    # the nine models: each the run that the plan names, one session, nine files of their own and none a published one
    known = {digest for files in weights.WEIGHT_SHA256.values() for digest in files.values()} | set(weights.START_SHA256.values())
    hashes = []
    for arm in ARMS:
        about = arms["systems"][arm]["system"]
        assert about["code"] == "07bf68e" and [model["seed"] for model in about["models"]] == [0, 1, 2]
        for model in about["models"]:
            assert model["identity"] == code.recipe_of(arm, model["seed"], ROOT / code.RECIPE) and model["sessions"] == 1
            hashes.append(model["sha256"])
        assert about["weights_sha256"] == [model["sha256"] for model in about["models"]]
    assert len(set(hashes)) == 9 and not known & set(hashes)
    said(ABLATIONS, "- **Each of the nine models was trained in one session.** No run went on from a checkpoint.",
         "The nine weight files are nine different files and none is a published one.")

    def number(system: str, name: str, part: str, key: str) -> float:
        return arms["systems"][system]["conditions"][name]["selection"][part][key]["value"]

    def f1(system: str, name: str = "unchanged") -> float:
        return number(system, name, "multi", "segment_f1_50")

    def apart(first: str, second: str, name: str = "unchanged", part: str = "multi", key: str = "segment_f1_50") -> dict:
        return arms["differences"][f"{first} minus {second}"][name][part][key]

    def right(system: str, name: str = "unchanged") -> int:
        count = SINGLE * number(system, name, "recordings", "accuracy")
        assert abs(count - round(count)) < 1e-6
        return round(count)

    # every difference of the file is the first value minus the second, and its interval is at the level the plan gives it
    conditions = list(arms["systems"][PUBLISHED]["conditions"])
    assert conditions == list(forward["systems"][PUBLISHED]["conditions"]) and len(conditions) == 27
    for first, second in code.DIFFERENCES:
        for name in conditions:
            for part, numbers in arms["differences"][f"{first} minus {second}"][name].items():
                for key, entry in numbers.items():
                    assert abs(entry["value"] - (number(first, name, part, key) - number(second, name, part, key))) < 1e-12 and entry["level"] == 95.0
    for system in code.SYSTEMS:
        for name in conditions[1:]:
            for part, numbers in arms["systems"][system]["conditions"][name]["minus_unchanged"].items():
                for key, entry in numbers.items():
                    assert abs(entry["value"] - (number(system, name, part, key) - number(system, "unchanged", part, key))) < 1e-12

    # the unchanged frames, and the rerun of the training document
    assert f"{f1(PUBLISHED):.4f}" == "0.9594" and f"{f1(AGAIN):.4f}" == "0.9565"
    ruled = arms["systems"][AGAIN]["unchanged_with"]["the setting by the rule"]
    assert f"{ruled['selection']['multi']['segment_f1_50']['value']:.4f}" == "0.9432" and ruled["setting"]["half_window"] == 5
    said("docs/TRAINING.md", "| Ensemble, decoded with the published setting | 0.9565 (0.9594) |",
         "| Ensemble, setting chosen again by the rule | 0.9432 (0.9594) |")
    said(ABLATIONS, "| unchanged | " + " | ".join(f3(f1(system)) for system in arms["systems"]) + " |",
         "| mirrored | " + " | ".join(f3(f1(system, "mirrored")) for system in arms["systems"]) + " |",
         f"the published system minus \"again\" is {spread(apart(PUBLISHED, AGAIN))}",
         "**The published system and the stage-2 ensemble were not run again.**")

    # the first question, from the intervals and without the tool
    family = arms["input_changes"]["conditions"]
    assert tuple(family) == FAMILY == code.FAMILY and arms["input_changes"]["level"] == 99.0

    def sentence(found: dict) -> str:
        return MOSTLY if found["interval"][0] > 0.5 else "mostly the loop without them" if found["interval"][1] < 0.5 else NOT_TOLD

    stands = {}
    for name, entry in family.items():
        first, second = sentence(entry["share"]), sentence(entry["with_the_published_system"]["share"])
        assert (first, second) == (entry["sentence"], entry["with_the_published_system"]["sentence"])
        stands[name] = first if first == second else None
        assert entry["stands"] == stands[name]
        again, without, stage2, published = (f1(system, name) for system in (AGAIN, WITHOUT, STAGE2, PUBLISHED))
        other = entry["with_the_published_system"]["share"]
        for found, given in ((entry["again_minus_without"], again - without), (entry["share"]["of"], again - stage2),
                             (entry["published_minus_again"], published - again), (entry["share"], (again - without) / (again - stage2)),
                             (other["of"], published - stage2), (other, (published - without) / (published - stage2))):
            assert abs(found["value"] - given) < 1e-12 and found["interval"][0] <= found["value"] <= found["interval"][1]  # what the values give
        assert {entry[key]["level"] for key in ("again_minus_without", "share", "published_minus_again")} == {99.0} == {entry["share"]["of"]["level"]}
    assert stands == {"noise 25": MOSTLY, "blur 2 pixels": MOSTLY, "64 pixels": NOT_TOLD, "speckle 0.25": None, "contrast 0.5": NOT_TOLD}
    said(ABLATIONS, "- **Speckle: the two trainings give two sentences, and neither stands.**",
         "- **The first question: right for three conditions, wrong for two.**")

    # the second question, from the differences and without the tool
    mirror = arms["mirror"]
    gain, here = apart(MIRROR, AGAIN, "mirrored"), apart(MIRROR, AGAIN)
    loss = arms["systems"][MIRROR]["conditions"]["mirrored"]["minus_unchanged"]["multi"]["segment_f1_50"]
    holds = {
        "gains under the mirrored image": gain["interval"][0] > 0,
        "stands the mirrored image": loss["value"] >= -0.03,
        "not below on the unchanged frames": not here["interval"][1] < 0 and here["value"] >= -0.01,
        "at most one recording fewer right": right(AGAIN) - right(MIRROR) <= 1,
        "not below under the five conditions": not any(apart(MIRROR, AGAIN, name)["interval"][1] < 0 for name in FAMILY),
    }
    assert holds == mirror["holds"] and list(holds.values()) == [True, False, False, True, False] and mirror["word"] == NOT_SUPPORTED == code.NOT_SUPPORTED
    assert (mirror["margins"]["its_own_loss"], mirror["margins"]["on_the_unchanged_frames"]) == (code.OWN_LOSS, code.UNCHANGED) == (0.03, 0.01)
    said(ABLATIONS, f"- **The word is \"{NOT_SUPPORTED}\".** Two of the five conditions hold and three do not.",
         f"| Under the mirrored image, the interval of \"with a mirror\" minus \"again\" lies wholly above zero | {spread(gain)} | yes |",
         f"| Under the mirrored image, the arm loses no more than 0.03 against its own unchanged frames | {spread(loss)} | no |",
         "- **The second question: wrong.** The word \"candidate\" was expected")
    below = [name for name in conditions if apart(MIRROR, AGAIN, name)["interval"][1] < 0]
    assert len(below) == 12 and below[0] == "unchanged"
    others = NUMERAL[len(below) - 1]  # the conditions beside the unchanged frames under which the interval lies below zero

    # the README and the model card: what part 4 measured and its verdict stand where the words "was not tested" stood
    limit = ("No training stage mirrored a frame. Three models trained again with mirrored frames reach "
             f"{f3(f1(MIRROR, 'mirrored'))} under the mirrored image, where the same recipe trained again without them reaches {f3(f1(AGAIN, 'mirrored'))}, "
             f"and they are lower on unchanged frames, {f3(f1(MIRROR))} against {f3(f1(AGAIN))}. By the rule that was fixed before they were trained "
             "they are no candidate to replace the published models, and they are not published ([docs/ABLATIONS.md](docs/ABLATIONS.md), section 13). "
             "When every window shows one frame sixteen times")
    for name in ("README.md", "MODEL_CARD.md"):
        said(name, limit)
        assert "was not tested" not in read(name)
    # the sentence about the agreement on MIMIC recordings, which part 4 did not score, says what it is about
    said("README.md", "which adds input augmentation and 1,500 updates; for this agreement the two were not told apart.")
    said("MODEL_CARD.md", "Stage 3 adds input augmentation and 1,500 updates, and for this agreement no run tells the two apart.")
    assert "no run tells the two apart" not in read("MODEL_CARD.md").replace("for this agreement no run tells the two apart.", "")
    assert "the two were not told apart" not in read("README.md").replace("for this agreement the two were not told apart.", "")
    said(ABLATIONS, "So the sentence was given the words \"for this agreement\". Its meaning is the one section 12 gave it, and its wording is changed "
         "where section 12 said that it stays.")

    # the changelog
    said("CHANGELOG.md",
         f"On the unchanged frames the recipe trained again has {f3(f1(AGAIN))} where the published system has {f3(f1(PUBLISHED))}, a difference that is not "
         "shown.",
         "Under noise with a spread of 25 grey levels and under a blur of 2 pixels the input changes are most of what stage 3 adds, with shares of "
         f"{f3(family['noise 25']['share']['value'])} and {f3(family['blur 2 pixels']['share']['value'])}.",
         "At 64 pixels and at a contrast of 0.5 the share is not told apart from one half, and for speckle two trainings of the recipe give two sentences, "
         "so that neither stands.",
         f"The arm with a mirror is \"{NOT_SUPPORTED}\". Under the mirrored image it has {f3(f1(MIRROR, 'mirrored'))} where the recipe trained again has "
         f"{f3(f1(AGAIN, 'mirrored'))}. On the unchanged frames it is {f3(-here['value'])} lower, and the interval of that difference lies below zero there "
         f"and under {others} other conditions.",
         "Of the plan's four expectations two held, the one about the input changes held for three of its five conditions, and the one about the mirror did "
         "not.",
         "No number contradicted the files except two ties of the last digit that had been rounded down: one accuracy of a training run (15,352 of "
         "16,000 windows, now 0.960) and one end of an interval of the cut recall (now -0.003).",
         "an Edit score of the stage-2 ensemble (+0.3 where +0.2 stood) and one end of an interval of the cut recall (+0.013 where +0.012 stood)")
    twice = apart(PUBLISHED, AGAIN)
    assert twice["interval"][0] < 0 < twice["interval"][1]  # "a difference that is not shown"
    # the accuracy that is a tie: 15,352 of the 16,000 windows of the last 500 updates
    tied = arms["systems"][WITHOUT]["system"]["models"][0]
    assert round(tied["training"]["accuracy"] * 16_000) == 15_352 and tied["identity"]["batch"] * 500 == 16_000 and f3(tied["training"]["accuracy"]) == "0.960"
    recall = arms["differences"][f"{MIRROR} minus {PUBLISHED}"]["noise 10"]["multi"]["cut_recall"]["interval"][1]
    assert (f"{recall:+.3f}", shown(recall, 3, True)) == ("-0.002", "-0.003")  # the other tie of section 13, as Python prints it and as it stands
    # the two cells of part 2 that were ties, as the file stores them
    edit = forward["systems"][STAGE2]["conditions"]["zoomed in 1.2"]["minus_unchanged"]["multi"]["edit"]["value"]
    end = forward["published_minus_stage2"]["rotated 10 degrees"]["multi"]["cut_recall"]["interval"][1]
    assert (f"{edit:+.1f}", shown(edit, 1, True), f"{end:+.3f}", shown(end, 3, True)) == ("+0.2", "+0.3", "+0.012", "+0.013")
    said(ABLATIONS, "the Edit score minus the unchanged frames is +0.3 where +0.2 stood", "the upper end of the interval of the cut recall is +0.013 where "
         "+0.012 stood")
