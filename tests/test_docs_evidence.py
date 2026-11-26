"""The data description and the numbers added on 2026-11-25 are held to the files they come from.

Nothing here runs a model: the definition files of the package and the stored outputs under docs/ are read, and the
documents must say what those files hold. A sentence that states a finding is looked up word for word, so that it
cannot be dropped or turned round without a test failing.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from importlib import resources
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VIEWS = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
ARMS = ("every 0.2 s", "every 0.1 s")


def definition(name: str) -> dict:
    return json.loads(gzip.decompress(resources.files("echoview.data").joinpath(name).read_bytes()))


def read(path: str) -> str:
    return (ROOT / path).read_text()


def stored(path: str) -> dict:
    return json.loads((DOCS / path).read_text())


def said(document: str, *phrases: str) -> None:
    """Every phrase stands in the document, word for word."""
    text = read(document)
    for phrase in phrases:
        assert phrase in text, f"{document} does not say: {phrase}"


@pytest.fixture(scope="module")
def test_sequences() -> list[dict]:
    return definition("ev9v_test.json.gz")["halves"]["test"]["sequences"]


@pytest.fixture(scope="module")
def development() -> dict[str, list[dict]]:
    return {half: body["sequences"] for half, body in definition("ev9v_dev.json.gz")["halves"].items()}


def keys(sequences: list[dict], kind: str = "native") -> set[str]:
    return {fragment["video"] for s in sequences if s["kind"] == kind for fragment in s["fragments"]}


# ------------------------------------------------------------------------------------------------ the splits
def test_no_recording_is_in_two_of_the_splits(test_sequences, development):
    recordings = definition("ev9v_recordings.json.gz")["recordings"]
    training = {r["key"] for r in recordings if r["split"] == "train"}
    parts = {"training": training, "calibration": keys(development["calibration"]), "selection": keys(development["selection"]),
             "test": keys(test_sequences)}
    assert {name: len(part) for name, part in parts.items()} == {"training": 3364, "calibration": 237, "selection": 284, "test": 800}
    names = list(parts)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not parts[first] & parts[second], f"{first} and {second} share a recording"
    for half, sequences in development.items():  # every fragment of a stream comes from a recording of its own split
        assert {f["video"] for s in sequences for f in s["fragments"]} <= parts[half]
    assert {f["video"] for s in test_sequences for f in s["fragments"]} <= parts["test"]
    assert len({s["group"] for s in test_sequences if s["kind"] == "native"}) == 125
    said("docs/DATA.md", "3,364 training recordings, 521 development recordings and 800 test recordings",
         "284 recordings in the `selection` half and 237", "800 recordings from 125 examinations",
         "The tests of this package check that no recording is in two of the definition files")


def test_the_counts_of_the_dataset_card_fit_what_the_project_uses(test_sequences, development):
    """The card's recordings per view and split, typed from the card as docs/DATA.md quotes it. Without the ninth
    code they must give the recordings per view that the definition files of this package hold."""
    card = {"PLHLA": (1120, 185, 275), "PASA": (212, 16, 41), "PMPALA": (319, 46, 88), "PMVLSA": (262, 32, 50), "PPMLSA": (183, 41, 56),
            "PMASA": (317, 50, 72), "A4C": (858, 148, 231), "A5C": (217, 29, 40), "SC4C": (195, 20, 35)}
    view_of = {"PLHLA": "PLAX", "PASA": "PSAX", "PMVLSA": "PSAX", "PPMLSA": "PSAX", "PMASA": "PSAX", "A4C": "A4C", "A5C": "A5C", "SC4C": "SC4C"}
    totals = [sum(counts[k] for counts in card.values()) for k in range(3)]
    assert totals == [3683, 567, 888] and sum(totals) == 5138
    expected = [Counter(), Counter(), Counter()]
    for code, view in view_of.items():
        for split in range(3):
            expected[split][view] += card[code][split]
    recordings = definition("ev9v_recordings.json.gz")["recordings"]
    held = [Counter(VIEWS[r["label"]] for r in recordings if r["split"] == "train"),
            Counter(VIEWS[s["fragments"][0]["label"]] for sequences in development.values() for s in sequences if s["kind"] == "native"),
            Counter(VIEWS[s["fragments"][0]["label"]] for s in test_sequences if s["kind"] == "native")]
    assert held == expected and [sum(c.values()) for c in held] == [3364, 521, 800]
    said("docs/DATA.md", "5,138 recordings", "3,683 recordings of 504 patients for training, 567 of 72 for validation and 888 of 127 for testing",
         "504, 72 and 127 sessions", "502, 72 and 125")


# -------------------------------------------------------------------------------------- how the streams are built
def test_the_streams_are_built_as_the_data_document_says(test_sequences, development):
    multi = [s for s in test_sequences if s["kind"] == "multi"]
    pairs = [s for s in test_sequences if s["kind"] == "pairs"]
    native = [s for s in test_sequences if s["kind"] == "native"]
    uses = Counter(f["video"] for s in multi for f in s["fragments"])
    joins = [(a["label"], b["label"]) for s in multi for a, b in zip(s["fragments"], s["fragments"][1:])]
    assert (sum(uses.values()), len(uses), max(uses.values())) == (701, 405, 11)
    assert (sum(a != b for a, b in joins), sum(a == b for a, b in joins)) == (379, 202)
    assert {len(s["fragments"]) for s in multi} == {4, 5, 6, 7, 8} and {len(s["fragments"]) for s in pairs} == {2}
    for s in multi + pairs:  # a recording is never twice in one stream, and a fragment is 1.0 to 4.0 s
        assert len({f["video"] for f in s["fragments"]}) == len(s["fragments"])
        assert all(10 <= len(f["frames"]) <= 40 for f in s["fragments"])
    assert max(Counter(f["video"] for s in pairs for f in s["fragments"]).values()) == 1  # used once in the pair streams
    assert Counter(s["fragments"][0]["label"] != s["fragments"][1]["label"] for s in pairs) == {True: 120, False: 120}
    by_view = Counter(VIEWS[f["label"]] for s in pairs for f in s["fragments"])
    assert (by_view["A5C"], by_view["SC4C"]) == (4, 0)
    # the change of grey levels: about half of the fragments of multi streams, the second fragment of half of the pair streams
    assert Counter(f["gamma"] for s in multi for f in s["fragments"]) == {0.9: 362, 1.0: 339}
    assert Counter((k, f["gamma"]) for s in pairs for k, f in enumerate(s["fragments"])) == {(0, 1.0): 240, (1, 1.0): 120, (1, 0.9): 120}
    assert {f["gamma"] for s in native for f in s["fragments"]} == {1.0}
    lengths = sorted(len(s["fragments"][0]["frames"]) for s in native)
    assert (lengths[0] / 10, lengths[-1] / 10) == (0.9, 53.5)
    assert lengths[len(lengths) // 2 - 1] / 10 <= 3.9 <= lengths[len(lengths) // 2] / 10
    exams = [s for sequences in development.values() for s in sequences if s["kind"] == "exam"]
    assert len(exams) == 72 and sum(len(s["fragments"]) - 1 for s in exams) == 449
    assert {f["gamma"] for s in exams for f in s["fragments"]} == {1.0}
    said("docs/DATA.md", "The 701 fragments of the test benchmark's multi streams come from 405 recordings, and one recording is used 11 times",
         "379 joins with a view change and 202 without", "120 joins with a view change and 120 without", "four A5C fragments and no SC4C fragment",
         "362 of the 701 in the test benchmark", "120 of 480", "of the 449 joins", "drawn uniformly from 10 to 40 samples",
         "lasts 0.9 to 53.5 s, and half of them at most 3.9 s", "A recording is never twice in one stream",
         "Whole recordings and examination streams are never changed")


# -------------------------------------------------------------------- the stored reports of `echoview evaluate`
def test_the_test_report_holds_the_published_numbers_and_the_confusion_table_of_the_documents(test_sequences):
    report = stored("evaluation/test_report.json")
    system, intervals = report["system"], report["intervals"]
    assert report["model"] == "C2_ens_v1" and report["sequences"] == {"native": 800, "pairs": 240, "multi": 120}

    def number(key: str) -> str:
        return f"{system[key]:.4f} (95% interval {intervals[key][0]:.3f} to {intervals[key][1]:.3f})"

    lines = [f"multi streams   segment F1@50 {number('segment_f1_50_multi')}", f"pair streams    segment F1@50 {number('segment_f1_50_pairs')}",
             f"recordings      accuracy {number('native_accuracy')}  macro-F1 {number('native_macro_f1')}"]
    said("docs/EVALUATION.md", *lines)  # what the command prints, as the document shows it
    said("README.md", lines[0])
    confusion = system["native_confusion"]
    assert [sum(row) for row in confusion] == [Counter(s["fragments"][0]["label"] for s in test_sequences if s["kind"] == "native")[k] for k in range(5)]
    assert sum(map(sum, confusion)) == 800 and sum(confusion[k][k] for k in range(5)) == 785
    for view, row in zip(VIEWS, confusion):
        right = 100 * row[VIEWS.index(view)] / sum(row)
        share = f"{right:.0f}%" if right == 100 else f"{right:.1f}%"
        said("docs/EVALUATION.md", f"| {view} | " + " | ".join(str(n) for n in row) + f" | {share} |")
    assert (confusion[2][3] + confusion[3][2], 800 - 785) == (12, 15)
    said("docs/EVALUATION.md", "Twelve of the fifteen errors are between A4C and A5C", "| Test benchmark | 800 | 15 | 14 |")
    said("MODEL_CARD.md", "Twelve of the fifteen single recordings that the test benchmark has wrong are between these two views")
    said("README.md", f"EchoRouter reads A4C as A5C in {confusion[2][3]} of the {sum(confusion[2])} single recordings of A4C")
    # the interval with streams drawn and the one with examinations drawn, wherever the 0.909 carries both
    row = stored("comparison/test.json")["rows"]["EchoRouter"]["multi"]["segment_f1_50"]
    streams = f"{intervals['segment_f1_50_multi'][0]:.3f} to {intervals['segment_f1_50_multi'][1]:.3f}"
    exams = f"{row['examinations'][0]:.3f} to {row['examinations'][1]:.3f}"
    both = f"0.909 (95% interval {streams} with streams drawn, {exams} with examinations drawn)"
    said("README.md", both, f"0.909 (95% interval {exams}, with whole examinations drawn)")
    said("MODEL_CARD.md", both)
    said("docs/EVALUATION.md", f"For segment F1 on the multi streams it gives {streams}.",
         f"over streams ({row['streams'][0]:.3f} to {row['streams'][1]:.3f}) and over the 125 examinations ({exams})")


def test_the_development_report_holds_the_calibration_and_the_examination_streams_of_the_documents():
    report = stored("evaluation/development_report.json")
    assert report["model"] == "C2_ens_v1"
    system = report["systems"]["ensemble_published_setting"]
    cal, sel = system["calibration"], system["selection"]
    assert f"{sel['system']['segment_f1_50_multi']:.4f}" == "0.9594" and f"{cal['system']['segment_f1_50_multi']:.4f}" == "0.9837"  # as published
    by_majority = [round(half["native"]["recordings"] * (1 - half["system"]["native_accuracy"])) for half in (cal, sel)]
    by_mean = [half["native"]["errors"] for half in (cal, sel)]
    assert (by_majority, by_mean) == ([2, 3], [2, 1])
    right = [100 * (1 - errors / half["native"]["recordings"]) for errors, half in zip(by_mean, (cal, sel))]
    exam = [half["streams_exam"] for half in (cal, sel)]
    assert [e["streams"] for e in exam] == [36, 36]
    assert round(cal["native"]["median_confidence"], 2) == round(sel["native"]["median_confidence"], 2) == 0.98
    said("docs/EVALUATION.md",
         f"| Development, calibration half | {cal['native']['recordings']} | {by_majority[0]} | {by_mean[0]} |",
         f"| Development, selection half | {sel['native']['recordings']} | {by_majority[1]} | {by_mean[1]} |",
         f"It is {cal['native']['ece']:.3f} on the calibration half and {sel['native']['ece']:.3f} on the selection half.",
         f"The median probability is 0.98 on both halves, and {right[0]:.1f}% and {right[1]:.1f}% of the recordings are right",
         f"segment F1 at 50% is {exam[0]['A6_f1']['50']:.3f} on the calibration half and {exam[1]['A6_f1']['50']:.3f} on the selection half, "
         f"and {100 * exam[0]['frame_error']:.1f}% and {100 * exam[1]['frame_error']:.1f}% of the samples get a wrong view",
         "(36 streams in each half", "Three recordings are wrong in all.")
    recordings = cal["native"]["recordings"] + sel["native"]["recordings"]
    misread = f"the first rule misreads {sum(by_majority)} of {recordings} recordings and the second {sum(by_mean)}"
    said("README.md", misread, f"segment F1 {exam[0]['A6_f1']['50']:.3f} and {exam[1]['A6_f1']['50']:.3f}")
    said("MODEL_CARD.md", misread,
         f"is {cal['native']['ece']:.3f} and {sel['native']['ece']:.3f}, with a median probability of 0.98 and three wrong recordings in all")


# ----------------------------------------------------------- the points added after the review of FOLLOWUP.md, section 9
def test_the_points_added_after_the_review_say_what_the_review_file_holds():
    review, result = stored("followup/confirmation_review.json"), stored("followup/confirmation.json")
    differences = result["differences_from_published"]
    at_radius = review["cut_distance_with_unmatched_changes_at_1s"]
    assert at_radius["every 0.1 s"] < at_radius["every 0.2 s"] < at_radius["EchoRouter"]
    both = {arm: entry["interval"] for arm, entry in review["cut_distance_both_units"].items()}
    assert all(interval[1] < 0 for interval in both.values())
    exact = review["recordings_exact_test"]
    assert [(exact[arm]["only_the_published_system"], exact[arm]["only_the_arm"], exact[arm]["p_two_sided"]) for arm in ARMS] == [(2, 1, 1.0), (1, 0, 1.0)]
    counted, floor = review["descriptions"], review["cut_distance_above_the_floor_of_its_grid"]
    f1 = differences["every 0.1 s"]["multi"]["segment_f1_50"]["interval"]
    assert f1[0] > 0
    said("docs/FOLLOWUP.md",
         f"the means are {at_radius['EchoRouter']:.3f} s for the published system, {at_radius['every 0.2 s']:.3f} s at 0.2 s and "
         f"{at_radius['every 0.1 s']:.3f} s at 0.1 s. The order is the same.",
         f"give [{both['every 0.2 s'][0]:.3f}, {both['every 0.2 s'][1]:.3f}] at 0.2 s and [{both['every 0.1 s'][0]:.3f}, {both['every 0.1 s'][1]:.3f}] "
         "at 0.1 s at the level of 97.5%",
         "so this interval is wider than either of the two by construction. (a) holds in it too. It is a check on the plan's interval, not a bootstrap",
         "Two recordings against one at 0.2 s, and one against none at 0.1 s", "gives p = 1.0 for both",
         f"holds {counted['differences_with_an_interval']} differences with a 95% interval, "
         f"and {counted['of_them_exclude_zero']} of those intervals exclude zero",
         "None is corrected for their number. Only (a) was a test.",
         f"Among its matched cuts the published system is {floor['EchoRouter']['matched_cuts']:.3f} s above the floor of its grid, the step of 0.2 s is "
         f"{floor['every 0.2 s']['matched_cuts']:.3f} s above its floor, and the step of 0.1 s is {floor['every 0.1 s']['matched_cuts']:.3f} s off",
         f"they are {floor['EchoRouter']['every_view_change']:.3f} s, {floor['every 0.2 s']['every_view_change']:.3f} s and "
         f"{floor['every 0.1 s']['every_view_change']:.3f} s above their floors",
         f"The interval of segment F1 at 0.1 s is [{f1[0]:+.4f}, {f1[1]:+.4f}]: it excludes zero at 95%",
         "end at exactly zero, for the reason given under (c). So does the lower end of the interval for the pair streams.",
         "**None of these points was planned, and none reads the test benchmark again.**")
    for arm in ARMS:  # the ends that are exactly zero
        assert differences[arm]["pairs"]["segment_f1_50"]["interval"][0] == 0.0
    recordings = differences["every 0.1 s"]["recordings"]
    assert recordings["macro_f1"]["interval"][1] == 0.0 and recordings["accuracy"]["interval"][1] == 0.0
    later = differences["every 0.2 s"]["recordings"]["macro_f1"]["interval"]
    assert later[0] < 0 < later[1]
    said("docs/FOLLOWUP.md", "with the interval [-0.015, +0.005], which includes zero", "That is also why the interval ends at exactly zero",
         "one includes zero and the other ends at it")
    passes = [result["confirmation"][arm]["forward_passes"] for arm in ARMS]
    said("docs/FOLLOWUP.md", f"the two steps read {passes[0]:.1f} and {passes[1]:.1f} times as many windows")
    said("docs/EVALUATION.md", f"are short and have {passes[0]:.1f} and {passes[1]:.1f} times as many")


def test_the_reworded_bullets_of_the_result_say_what_the_read_holds():
    result = stored("followup/confirmation.json")
    differences, rows = result["differences_from_published"], result["rows"]
    a5c = [differences[arm]["recordings"]["recall_A5C"] for arm in ARMS]
    assert all(entry["interval"][1] == 0.0 for entry in a5c)  # "both intervals end at zero"

    def points(entry: dict) -> str:
        return f"{-100 * entry['value']:.1f} points [{100 * entry['interval'][0]:.1f}, {100 * entry['interval'][1]:.1f}]"

    said("docs/FOLLOWUP.md", f"the recall of A5C is lower by {points(a5c[0])} and by {points(a5c[1])}, and both intervals end at zero",
         "With one to three recordings changing, the read shows no direction for it in single recordings")
    gains = {view: differences["every 0.1 s"]["multi"][f"recovery_{view}"] for view in VIEWS}
    assert all(entry["interval"][0] > 0 for entry in gains.values())
    others = sorted(100 * gains[view]["value"] for view in ("PLAX", "PSAX", "SC4C"))
    recovered = {view: (100 * rows["every 0.1 s"]["multi"][f"recovery_{view}"]["value"], 100 * rows["EchoRouter"]["multi"][f"recovery_{view}"]["value"])
                 for view in ("A4C", "A5C")}
    for view in ("A4C", "A5C"):
        middle = differences["every 0.1 s"]["multi"][f"piece_midpoint_{view}"]["interval"]
        assert middle[0] < 0 < middle[1]
    said("docs/FOLLOWUP.md",
         f"A5C {recovered['A5C'][0]:.1f}% against {recovered['A5C'][1]:.1f}% and A4C {recovered['A4C'][0]:.1f}% against {recovered['A4C'][1]:.1f}%, "
         f"and PLAX, PSAX and SC4C by {others[0]:.1f} to {others[-1]:.1f} points",
         "Each of the five 95% intervals lies above zero, and none is corrected for their number.",
         "the middles of the A4C and the A5C pieces are not shown to change")
    development = stored("followup/followup.json")["differences_from_published"]
    nearer = []
    for arm in ("every 0.1 s", "every 0.2 s"):
        test, dev = differences[arm]["multi"]["cut_distance_mean_s"], development[f"{arm}, own setting"]["multi"]["cut_distance_mean_s"]
        assert test["interval"][0] < dev["interval"][1] and dev["interval"][0] < test["interval"][1]
        nearer.append([f"{-e['value']:.3f} s [{e['interval'][0]:.3f}, {e['interval'][1]:.3f}]" for e in (test, dev)])
    said("docs/FOLLOWUP.md",
         f"The cuts come nearer by {nearer[0][0]} and {nearer[1][0]}, where the selection half gave {nearer[0][1]} and {nearer[1][1]}",
         "their intervals overlap, and the difference between them was not tested")
    cost = [differences[arm]["recordings"]["macro_f1"] for arm in ARMS]
    said("README.md",
         f"lower by {-cost[0]['value']:.3f} (95% interval {cost[0]['interval'][0]:.3f} to {cost[0]['interval'][1]:+.3f}) and by "
         f"{-cost[1]['value']:.3f} ({cost[1]['interval'][0]:.3f} to {cost[1]['interval'][1]:.3f}), so the read does not show whether that cost is real")


def test_the_corrections_of_the_comparison_say_what_its_output_holds():
    output = stored("comparison/addendum/test.json")
    differences, rows = output["differences_from_echorouter"], output["rows"]
    middle = {name: differences[name]["multi"]["piece_midpoint_A5C"] for name in ("EchoViewCLIP", "STFM", "ASFormer", "EchoPrime + decoder")}
    for name in ("EchoViewCLIP", "STFM"):
        assert middle[name]["interval"][0] < 0 < middle[name]["interval"][1]
    assert middle["ASFormer"]["interval"][0] > 0 and middle["EchoPrime + decoder"]["interval"][0] > 0
    shown = {name: f"{100 * e['value']:.1f} points, {100 * e['interval'][0]:.1f} to {100 * e['interval'][1]:.1f}" for name, e in middle.items()}
    compared = [name for name in differences if not name.startswith(("EchoRouter", "Perfect")) and "released numbers" not in name]
    f1 = {name: differences[name]["multi"]["segment_f1_50_A5C"]["interval"] for name in compared}
    assert sorted(name for name, interval in f1.items() if interval[0] <= 0) == ["ASFormer + decoder"]
    grid = differences["Perfect on the 0.5 s grid"]["multi"]
    distance, near = grid["cut_distance_mean_s"], grid["cut_recall_0.1s"]
    matched = rows["EchoRouter"]["multi"]["cut_distance_mean_s"]["support"], rows["EchoRouter"]["multi"]["cut_matched_share"]["support"]
    said("docs/COMPARISON.md",
         f"the differences to EchoViewCLIP ({shown['EchoViewCLIP']}) and to STFM ({shown['STFM']}) include zero; those to ASFormer and to EchoPrime's "
         "classifier do not",
         "lies above zero for every row as published or with a decoder setting of its own, except ASFormer with the decoder",
         f"EchoRouter's matched cuts are {distance['value']:.3f} s further off than a perfect system's on that grid "
         f"({distance['interval'][0]:.3f} to {distance['interval'][1]:.3f}), and {-100 * near['value']:.1f} points fewer of the view changes have a cut "
         f"within 0.1 s ({-100 * near['interval'][1]:.1f} to {-100 * near['interval'][0]:.1f})",
         f"EchoRouter has a matched cut for {matched[0]:.0f} of the {matched[1]:.0f} view changes")
    wrong = {name: int(rows[name]["recordings"]["a4c_as_a5c"]["value"]) for name in [*compared, "EchoRouter"]}
    others = sorted(wrong[name] for name in compared)
    assert (others[0], others[-1], wrong["STFM + decoder"], wrong["EchoRouter"]) == (1, 8, 8, 10)
    said("README.md", f"the others in {others[0]} to {others[-1]}.")
    said("docs/COMPARISON.md", f"STFM with the decoder, a row of the addendum, does so in {wrong['STFM + decoder']}.")
    accuracy = stored("comparison/test.json")["differences_from_echorouter"]["EchoViewCLIP"]["recordings"]["accuracy"]
    said("docs/FOLLOWUP.md",
         f"the 95% interval of the difference, {-100 * accuracy['interval'][1]:.1f} to {-100 * accuracy['interval'][0]:.1f} points, includes zero")


# ------------------------------------------------------------------------------------------------- the speed
def test_the_speed_table_says_what_the_measurements_hold():
    l40s, wide = stored("speed/nvidia_l40s.json"), stored("speed/nvidia_l40s_batch32.json")
    for report in (l40s, wide):
        assert report["reference_system"] is True and report["stream_seconds"] == 60.0 and report["environment"]["device_name"] == "NVIDIA L40S"
        assert [step["runs"] for step in report["steps"]] == [5, 5, 5]
    assert (l40s["environment"]["batch"], wide["environment"]["batch"]) == (8, 32)
    names = ("every 0.5 s (published)", "every 0.2 s", "every 0.1 s")
    for name, step in zip(names, l40s["steps"]):
        said("docs/EVALUATION.md",
             f"| {name} | {step['windows']} | {step['seconds_median']:.2f} ({step['seconds_min']:.2f} to {step['seconds_max']:.2f}) | "
             f"{step['times_real_time']:.1f} | {step['times_the_published_step']:.1f} |")
    seconds = [step["seconds_median"] for step in l40s["steps"]]
    fast = [step["times_real_time"] for step in l40s["steps"]]
    per_window = [1000 * step["seconds_median"] / step["windows"] for step in l40s["steps"]]
    said("docs/EVALUATION.md",
         f"PyTorch held at most {l40s['gpu_memory_peak_gib']:.1f} GiB of GPU memory for the 60 s stream",
         f"the three steps take {wide['steps'][0]['seconds_median']:.2f}, {wide['steps'][1]['seconds_median']:.2f} and "
         f"{wide['steps'][2]['seconds_median']:.2f} s and hold {wide['gpu_memory_peak_gib']:.1f} GiB",
         f"A window cost {per_window[0]:.1f}, {per_window[1]:.1f} and {per_window[2]:.1f} ms",
         f"with PyTorch {l40s['environment']['torch'].split('+')[0].rsplit('.', 1)[0]}")
    assert all(wide_step["seconds_median"] > step["seconds_median"] for wide_step, step in zip(wide["steps"], l40s["steps"]))  # "no faster"
    sentence = (f"the published system takes {seconds[0]:.1f} s, the step of 0.2 s takes {seconds[1]:.1f} s and the step of 0.1 s takes {seconds[2]:.1f} s: "
                f"{fast[0]:.1f}, {fast[1]:.1f} and {fast[2]:.1f} times as fast as the stream plays")
    said("docs/FOLLOWUP.md", sentence)
    said("README.md",
         f"works through a stream of 60 s in {seconds[0]:.1f} s, which is {fast[0]:.1f} times as fast as the stream plays, and PyTorch holds "
         f"{l40s['gpu_memory_peak_gib']:.1f} GiB of GPU memory for it. A read-out every 0.2 s takes {seconds[1]:.1f} s and one every 0.1 s "
         f"takes {seconds[2]:.1f} s")
    said("CHANGELOG.md",
         f"a stream of 60 s takes {seconds[0]:.1f} s, {seconds[1]:.1f} s and {seconds[2]:.1f} s, with at most {l40s['gpu_memory_peak_gib']:.1f} GiB")


def test_the_read_for_classify_is_what_the_documents_say():
    """docs/evaluation/classify_test.json is the output of the read of 2026-11-26 and classify_trial.json that of its
    trial. The sentences of the result that state a number are built here from those two files and looked up word for
    word. Times, and what the result says about the plan's expectations, have no file to be held to."""
    from echoview.evaluation import classify_read, comparison

    doc = "docs/EVALUATION.md"
    result, trial = stored("evaluation/classify_test.json"), stored("evaluation/classify_trial.json")
    digest = comparison.file_sha256(DOCS / "evaluation" / "classify_test.json")
    assert classify_read.READ_SHA256 == digest
    assert result["sha256"]["trial"] == comparison.file_sha256(DOCS / "evaluation" / "classify_trial.json")
    assert (result["benchmark"], result["sequences"], result["draws"], result["seed"]) == ("test", 1160, 10_000, 20261123)
    assert result["first_read_rows_reproduced"] == ["EchoRouter", "Perfect on the 0.5 s grid"]
    assert result["word"] == classify_read.NOT_SHOWN == "not shown to differ from the majority rule"
    said(doc, f"SHA-256 `{digest}`", "with the code of commit `c7b1c3d`. `classify`'s rule is not shown to differ from the majority rule.**",
         "**Status: planned and read on 2026-11-26 (UTC). `classify`'s rule is not shown to differ from the majority rule.**")
    majority, mean = result["rules"][classify_read.MAJORITY], result["rules"][classify_read.CLASSIFY]
    assert majority == stored("comparison/test.json")["rows"]["EchoRouter"]["recordings"]  # the published rule, as the comparison holds it
    found, probability = result["classify"], result["classify"]["probability"]
    assert (found["recordings"], found["errors"], found["units_drawn"], found["units_with_a_wrong_recording"]) == (800, 14, 125, 14)
    said(doc, "**The 14 wrong recordings lie in 14 of the 125 examinations.**")

    def share(number: dict) -> str:
        return f"{100 * number['value']:.1f}% [{100 * number['interval'][0]:.1f}, {100 * number['interval'][1]:.1f}]"

    def plain(number: dict) -> str:
        return f"{number['value']:.3f} [{number['interval'][0]:.3f}, {number['interval'][1]:.3f}]"

    def with_interval(number: dict, signed: bool = False) -> str:
        form = "+.3f" if signed else ".3f"
        return f"{number['value']:{form}}, with the interval [{number['interval'][0]:{form}}, {number['interval'][1]:{form}}]"

    # the view
    rows = []
    for name, rule, right in (("Majority of the decoded samples (published)", majority, 785), ("Largest mean probability (`classify`)", mean, 786)):
        assert round(800 * rule["accuracy"]["value"]) == right and rule["recall_PLAX"]["value"] == rule["recall_PSAX"]["value"] == 1.0
        as_a5c, as_a4c = rule["a4c_as_a5c"], rule["a5c_as_a4c"]
        assert as_a4c["interval"] is None  # a count of 5 or fewer has no interval
        cells = [name, str(right), share(rule["accuracy"]), plain(rule["macro_f1"]), share(rule["recall_A4C"]), share(rule["recall_A5C"]),
                 share(rule["recall_SC4C"]), f"{as_a5c['value']:.0f} [{as_a5c['interval'][0]:.0f}, {as_a5c['interval'][1]:.0f}]", f"{as_a4c['value']:.0f}"]
        rows.append("| " + " | ".join(cells) + " |")
    said(doc, "\n".join(rows), "Both rules read every recording of PLAX and of PSAX right.")
    table, published = found["confusion"], result["confusion_majority"]
    assert sum(map(sum, table)) == 800 and sum(table[k][k] for k in range(5)) == 786
    assert published == stored("evaluation/test_report.json")["system"]["native_confusion"]
    said(doc, "| | PLAX | PSAX | A4C | A5C | SC4C |\n|---|---|---|---|---|---|\n"
         + "\n".join(f"| {view} | " + " | ".join(str(n) for n in row) + " |" for view, row in zip(VIEWS, table)))
    assert (table[2][3] + table[3][2], published[2][3] + published[3][2]) == (11, 12) and (table[2][2], published[2][2]) == (220, 219)
    said(doc, "11 of the 14 recordings that `classify`'s rule has wrong are between A4C and A5C, as 12 of the majority rule's 15 are",
         "`classify`'s rule reads one recording of A4C more right, 220 against 219")

    # the two rules against each other
    difference, only = result["classify_minus_majority"], result["recordings_only_one_rule_reads_right"]
    accuracy, recall = difference["accuracy"], difference["recall_A5C"]
    assert only == {"only_classify": 2, "only_the_majority": 1, "p_two_sided": 1.0} and accuracy["interval"][0] < 0 < accuracy["interval"][1]
    low, high = (100 * end for end in accuracy["interval"])
    said(doc, "It reads 786 of the 800 recordings right, and the majority rule reads 785.",
         f"The difference in accuracy is {100 * accuracy['value']:.1f} points, with the interval [{low:+.1f}, {high:+.1f}].",
         "Two recordings are read right only by `classify`'s rule and one only by the majority rule, and the exact test of that split gives p = 1.0.",
         "Neither of the two things that the word needs holds.",
         f"is 0.959 against 0.956: {with_interval(difference['macro_f1'], signed=True)}.")
    assert recall["value"] == 0.0 and table[3] == published[3]
    low, high = (100 * end for end in recall["interval"])
    said(doc, f"the interval of the difference in the recall of A5C is not zero ([{low:+.1f}, {high:+.1f}] points)")
    of_a4c, fewer = difference["recall_A4C"], difference["a4c_as_a5c"]
    assert of_a4c["interval"][0] == 0.0 and fewer["interval"][1] == 0.0 and fewer["value"] == -1.0  # both end at exactly zero
    low, high = (100 * end for end in of_a4c["interval"])
    said(doc, f"The difference in the recall of A4C is {100 * of_a4c['value']:+.1f} points, with the interval [{low:+.1f}, {high:+.1f}], "
              f"and A4C is read as A5C in one recording fewer, with the interval [{fewer['interval'][0]:.0f}, {fewer['interval'][1]:.0f}]. "
              "Both intervals end at exactly zero")
    assert difference["a5c_as_a4c"]["value"] == 0.0 and difference["a5c_as_a4c"]["interval"] is None

    # the probability
    bins, parts = [b for b in probability["bins"] if b["recordings"]], probability["bins_equal_count"]
    lines = [f"| above {b['above']:.1f}, up to {b['up_to']:.1f} | {b['recordings']} | {b['right']} | {b['mean_probability']:.3f} "
             f"| {100 * b['share_right']:.1f}% |" for b in bins]
    said(doc, "| Probability | Recordings | Right | Mean probability | Share that is right |\n|---|---|---|---|---|\n" + "\n".join(lines)
         + "\n\nNo recording has a probability of 0.4 or less.")
    assert [b["recordings"] for b in probability["bins"][:4]] == [0, 0, 0, 0] and bins[0]["above"] == 0.4
    assert all(b["share_right"] >= b["mean_probability"] for b in bins + parts)  # every gap points the same way
    gap = probability["probability_minus_right"]
    assert f"{probability['ece']:.3f}" == f"{probability['ece_equal_count']:.3f}" == f"{-gap['value']:.3f}" == "0.025"
    assert (f"{probability['mean_probability']:.3f}", round(800 * probability["share_right"])) == ("0.958", 786)
    assert f"{100 * 786 / 800:.2f}" == "98.25"
    said(doc, "The mean probability is 0.958, and 786 of the 800 recordings are right (98.25%).",
         f"The mean probability minus the share that is right is {with_interval(gap)}.",
         "**The two calibration errors are that gap.** Both are 0.025. "
         "In every bin and in every part of equal count the share that is right is at least the mean probability")
    assert [b["recordings"] for b in parts] == [80] * 10 and [b["right"] for b in parts] == [67, 79] + [80] * 8
    said(doc, f"In the tenth with the lowest probabilities, from {parts[0]['from']:.3f} to {parts[0]['to']:.3f}, 67 of the 80 recordings are right, "
              f"at a mean probability of {parts[0]['mean_probability']:.3f}. In the next tenth, up to {parts[1]['to']:.3f}, 79 of 80 are right. "
              f"In the other eight tenths, with probabilities from {parts[2]['from']:.3f} to {parts[9]['to']:.3f}, all 640 are right.")
    said(doc, f"The Brier score is {with_interval(probability['brier'])}.", f"The log loss is {with_interval(probability['log_loss'])}.")
    separation = probability["separation"]
    assert separation["draws_without_a_number"] == 0
    assert f"{probability['median_probability_right']:.3f}" == f"{probability['median_probability']:.3f}" == "0.980"
    said(doc, f"The median probability is {probability['median_probability']:.3f} for all recordings and for the right ones, "
              f"and {probability['median_probability_wrong']:.3f} for the 14 wrong ones.",
         f"The separation is {with_interval(separation)}: a recording that is read right has the higher probability in "
         f"{100 * separation['value']:.0f}% of the pairs with one that is read wrong. Every draw of examinations held a wrong recording.")
    wrong = {b["up_to"]: b["recordings"] - b["right"] for b in bins}
    below = [b for b in bins if b["up_to"] < 0.95]
    assert sum(n for top, n in wrong.items() if top < 0.85) == 12 and wrong[1.0] == 2
    assert (sum(b["recordings"] for b in below), sum(b["right"] for b in below)) == (52, 40)
    said(doc, "12 of the 14 wrong recordings have a probability of 0.8 at most, and two have one above 0.9. "
              "Of the 52 recordings with a probability of 0.9 at most, 40 are right.", "By rule 3 no threshold is chosen or proposed from it.")
    assert found["seeds"] == {"recordings_where_the_seeds_differ": 1, "of_them_wrong": 0, "wrong_where_the_seeds_agree": 14}
    said(doc, "The three seeds name different views in one recording of the 800, and that one is read right. "
              "In all 14 wrong recordings the three seeds name the same view.")

    # the trial: the development report's numbers, as the documents quote them
    report = stored("evaluation/development_report.json")["systems"][classify_read.TRIAL_SYSTEM]
    assert classify_read.trial_held(trial)
    assert trial["sha256"]["report"] == comparison.file_sha256(DOCS / "evaluation" / "development_report.json")
    cal, sel = (trial["numbers_reproduced"][half] for half in ("calibration", "selection"))
    for got, half in ((cal, "calibration"), (sel, "selection")):
        known = report[half]["native"]
        assert (got["recordings"], got["errors"], got["confusion"]) == (known["recordings"], known["errors"], known["confusion"])
        assert abs(got["ece"] - known["ece"]) < 1e-12 and abs(got["median_probability"] - known["median_confidence"]) < 1e-12
    said(doc, f"{cal['errors']} and {sel['errors']} wrong recordings of {cal['recordings']} and {sel['recordings']}, the two confusion tables, "
              f"calibration errors of {cal['ece']:.4f} and {sel['ece']:.4f} and median probabilities of "
              f"{cal['median_probability']:.4f} and {sel['median_probability']:.4f}")

    # the other documents
    of_wrong, of_right = f"{probability['median_probability_wrong']:.2f}", f"{probability['median_probability_right']:.2f}"
    for document, there in (("README.md", ""), ("MODEL_CARD.md", " there")):
        said(document, "On the test benchmark the first misreads 15 of 800 and the second 14. "
                       "One read that was planned beforehand does not show the two rules to differ there",
             "786 of the 800 recordings are right (98.25%)", f"mean probability is {probability['mean_probability']:.3f}",
             f"The 14 wrong recordings{there} have a median probability of {of_wrong}")
    said("MODEL_CARD.md", f"the calibration error is {probability['ece']:.3f}", f"has the higher probability in {100 * separation['value']:.0f}% of the pairs")
    said(doc, "On the selection half `classify`'s rule has two wrong recordings fewer, and on the test benchmark one fewer.",
         f"gives a calibration error of {probability['ece']:.3f}", f"have a median probability of {of_wrong}, the right ones {of_right}",
         "| 2026-11-26 | One read of stored outputs. `classify`'s rule is scored on the 800 single recordings; the streams are decoded only to hold",
         "Three recordings are read right by only one of the two rules, where at most five were expected",
         "They lie in 14 different ones.", "Every number about wrong answers rests on 14 recordings, 11 of them between A4C and A5C.",
         f"The plan expected a wide interval; it is [{separation['interval'][0]:.3f}, {separation['interval'][1]:.3f}].")
    reproduced, reported = result["published_numbers_reproduced"], stored("evaluation/test_report.json")["system"]
    assert reproduced == {key: reported[key] for key in reproduced} and len(reproduced) == 4
    said("MODEL_CARD.md", "once for the denser read-out (2026-11-25) and once for what `classify` returns (2026-11-26)")
    said("CHANGELOG.md", "The largest mean probability reads 786 recordings right, where the majority of the decoded samples, "
                         "the rule of the published numbers, reads 785.",
         "the mean probability is 0.958, and the calibration error is 0.025",
         f"have a median probability of {of_wrong} and the right ones {of_right}, and the three seeds agree on every wrong one")


# ---------------------------------------------------------------------------------------------------------------------
# docs/ablations/stored.json is the output of part 1 of docs/ABLATIONS.md. The tables of the document are built here
# from that file, row by row and with their header rows, the sentences that state a number are looked up word for
# word, and what a sentence says without a number (highest, below, none) is asserted from the file beside it.

ABLATIONS = "docs/ABLATIONS.md"
SINGLE = 284  # the single recordings of the selection half
WHOLE = "seeds 0+1+2"


def f3(x: float) -> str:
    return f"{x:.3f}"


def spread(entry: dict, digits: int = 3, scale: float = 1.0) -> str:
    """A difference with its interval, as the tables of differences give it."""
    if entry.get("interval") is None:
        return f"{scale * entry['value']:+.{digits}f}"
    low, high = entry["interval"]
    return f"{scale * entry['value']:+.{digits}f} [{scale * low:+.{digits}f}, {scale * high:+.{digits}f}]"


def apart(entry: dict) -> bool:
    """The interval lies wholly on one side of zero."""
    return entry.get("interval") is not None and (entry["interval"][0] > 0 or entry["interval"][1] < 0)


def whole(x: float) -> int:
    assert abs(x - round(x)) < 1e-6, x
    return round(x)


def right(selection: dict) -> int:
    assert selection["recordings"]["accuracy"]["support"] == SINGLE
    return whole(SINGLE * selection["recordings"]["accuracy"]["value"])


def test_part_one_of_the_ablations_the_run_and_the_size_of_the_ensemble():
    from echoview.evaluation import comparison

    d = stored("ablations/stored.json")
    assert (d["benchmark"], d["draws"], d["seed"]) == ("development", 10_000, 20261123) and d["sequences"] == {"calibration": 472, "selection": 541}
    assert f"{d['published_selection_f1']:.4f}" == "0.9594" == f"{comparison.PUBLISHED_SELECTION_F1:.4f}"
    said(ABLATIONS, "gave the published number of the selection half, 0.9594", "with the code of commit `fb674f6`",
         "Intervals are at 95% unless a line says 99%. Streams are drawn for the numbers about streams and recordings for the numbers about recordings.",
         "**The text of part 1 was corrected on 2026-11-26, after an independent check of it against the file.** No number was wrong.")

    systems = d["ensemble_size"]["systems"]
    assert list(systems) == ["seed 0", "seed 1", "seed 2", "seeds 0+1", "seeds 0+2", "seeds 1+2", WHOLE]
    rows = ["| System | Decoder setting by the rule | Segment F1 at 50%, multi streams | Minus the three seeds together | Pair streams | Recordings "
        "right of 284, "
            "majority rule | Wrong by `classify`'s rule | Calibration error, `classify`'s rule |", "|---|---|---|---|---|---|---|---|"]
    for name, e in systems.items():
        s, m = e["selection"], e["minus_the_whole_ensemble"]["multi"]["segment_f1_50"]
        f1 = s["multi"]["segment_f1_50"]
        rows.append(f"| {name} | half window {e['setting']['half_window']}, threshold {e['setting']['threshold']:.3f} | {f3(f1['value'])} "
                    f"[{f3(f1['interval'][0])}, {f3(f1['interval'][1])}] | {'' if name == WHOLE else spread(m)} | {f3(s['pairs']['segment_f1_50']['value'])} "
                    f"| {right(s)} | {e['classify']['errors']} | {f3(e['classify']['ece'])} |")
    said(ABLATIONS, "\n".join(rows))

    # on the multi streams the three together are highest; three intervals lie below zero, at 95%
    smaller = {name: e for name, e in systems.items() if name != WHOLE}
    f1 = {name: e["selection"]["multi"]["segment_f1_50"]["value"] for name, e in systems.items()}
    assert max(f1, key=f1.get) == WHOLE
    minus = {name: e["minus_the_whole_ensemble"]["multi"]["segment_f1_50"] for name, e in smaller.items()}
    assert all(m["level"] == 95.0 for m in minus.values())
    assert [name for name, m in minus.items() if m["interval"][1] < 0] == ["seed 1", "seed 2", "seeds 1+2"]
    by_size = d["ensemble_size"]["mean_segment_f1_50_multi_by_size"]
    said(ABLATIONS, "**On the multi streams the three seeds together have the highest segment F1.** "
                    f"The mean over the single seeds is {f3(by_size['1'])}, over the pairs {f3(by_size['2'])}, and the three together reach "
                    f"{f3(by_size['3'])}. "
                    "Against seed 1, seed 2 and their pair the interval of the difference lies below zero. Against seed 0 and the two pairs that "
                    "hold it, it does not: "
                    f"seed 0 alone is {f3(-minus['seed 0']['value'])} below the three together, with an interval from {minus['seed 0']['interval'][0]:+.3f} to "
                    f"{minus['seed 0']['interval'][1]:+.3f}. These are six intervals at 95%, and none is corrected for their number.")

    # on the other numbers they are not
    pairs = {name: e["selection"]["pairs"]["segment_f1_50"]["value"] for name, e in systems.items()}
    assert [name for name in systems if pairs[name] > pairs[WHOLE] + 1e-12] == ["seed 1", "seed 2"]
    counts = {name: right(e["selection"]) for name, e in systems.items()}
    assert [name for name in systems if counts[name] > counts[WHOLE]] == ["seed 2", "seeds 0+2", "seeds 1+2"] and set(counts.values()) == {281, 282}
    every = {name: e["minus_the_whole_ensemble"] for name, e in smaller.items()}
    assert not any(apart(m["pairs"]["segment_f1_50"]) or apart(m["recordings"]["accuracy"]) for m in every.values())
    said(ABLATIONS, "**On pair streams, and on single recordings by the majority rule, they are not the highest.** "
                    f"On pair streams seed 1 ({f3(pairs['seed 1'])}) and seed 2 ({f3(pairs['seed 2'])}) are above the three together ({f3(pairs[WHOLE])}). "
                    f"By the majority rule the three together read {counts[WHOLE]} of the {SINGLE} single recordings right, and seed 2 and the two "
                    f"pairs that hold it read {counts['seed 2']}. No interval of these differences excludes zero.")

    # every smaller system minus the three together, for every number, and where the three together stand
    said(ABLATIONS, "\n".join(["| Minus the three seeds together | " + " | ".join(smaller) + " |", "|---|---|---|---|---|---|---|"]
                              + [f"| {title} | " + " | ".join(spread(every[name][part][key], digits, scale) for name in smaller) + " |"
                                 for title, part, key, digits, scale in COMPARED_ROWS]))
    assert all(entry["level"] == 95.0 for m in every.values() for part in m.values() for entry in part.values())
    lower_is_better = ("wrong_samples", "cut_distance_mean_s", "same_view_cut_rate", "extra_segments", "missed_segments")

    def stands(key: str) -> str:
        sign = -1.0 if key in lower_is_better else 1.0
        values = [sign * e["selection"]["multi"][key]["value"] for e in systems.values()]
        mine = sign * systems[WHOLE]["selection"]["multi"][key]["value"]
        return "not" if mine < max(values) else "alone" if values.count(max(values)) == 1 else "level"

    assert {key: stands(key) for key in systems[WHOLE]["selection"]["multi"]} == {
        "segment_f1_50": "alone", "edit": "alone", "wrong_samples": "alone", "cut_recall": "level", "missed_segments": "level", "extra_segments": "level",
        "same_view_cut_rate": "level", "cut_precision": "not", "cut_distance_mean_s": "not"}
    intervals = sum(m[part][key].get("interval") is not None for m in every.values() for _, part, key, _, _ in COMPARED_ROWS)
    said(ABLATIONS, "**Of the other numbers on the multi streams** the three together are alone the best on the Edit score and on wrong samples. "
                    "They are level with the best on cut recall, segments missed, segments that match nothing and cuts at joins of the same view. "
                    "They are not the best on cut precision and on the distance of the matched cuts. The table above has every difference. "
                    f"Its {intervals} intervals are at 95%, and none is corrected for their number.")

    # the rule gives the three seeds the published setting, and no smaller system
    assert [name for name, e in systems.items() if e["is_the_published_setting"]] == [WHOLE]
    published = d["decoders"]["released"]["setting"]
    assert systems[WHOLE]["setting"]["half_window"] == published["half_window"] == 7
    assert abs(systems[WHOLE]["setting"]["threshold"] - published["threshold"]) < 1e-4 and systems[WHOLE]["selection"] == d["decoders"]["released"]["selection"]
    assert all(abs(e["setting"]["threshold"] - published["threshold"]) > 1e-3 for e in smaller.values())
    assert sum(e["setting"]["half_window"] != 7 for e in smaller.values()) == 4
    said(ABLATIONS, "half window 7, a threshold that agrees with the published one to 0.0001, and the same value for every number. "
                    "Every smaller system gets another threshold, and four of the six another half window.")

    eces = sorted(e["classify"]["ece"] for e in systems.values())
    assert {e["classify"]["errors"] for e in systems.values()} == {1}
    said(ABLATIONS, f"Every system has one wrong recording of {SINGLE}, and the calibration errors lie between {f3(eces[0])} and {f3(eces[-1])}.")
    below = [-m["value"] for m in minus.values()]
    assert min(below) > 0  # above each on the multi streams
    said(ABLATIONS, f"On the multi streams they are above each, by {f3(max(below))} at most. On pair streams and by the majority rule they are not above each.",
         f"From the published logits and by the rule they are {f3(f1['seeds 0+2'])} and {f3(f1['seeds 1+2'])}.")


COMPARED_ROWS = (  # the row's name in the table of differences, the part, the key, the digits, the scale
    ("Segment F1 at 50%, multi streams", "multi", "segment_f1_50", 3, 1.0),
    ("Edit", "multi", "edit", 1, 1.0),
    ("Wrong samples, percentage points", "multi", "wrong_samples", 1, 100.0),
    ("Cut precision", "multi", "cut_precision", 3, 1.0),
    ("Cut recall", "multi", "cut_recall", 3, 1.0),
    ("Mean distance of matched cuts, seconds", "multi", "cut_distance_mean_s", 3, 1.0),
    ("Cuts at joins of the same view, percentage points", "multi", "same_view_cut_rate", 1, 100.0),
    ("Segments that match nothing", "multi", "extra_segments", 0, 1.0),
    ("Segments missed", "multi", "missed_segments", 0, 1.0),
    ("Segment F1 at 50%, pair streams", "pairs", "segment_f1_50", 3, 1.0),
    ("Accuracy, single recordings", "recordings", "accuracy", 3, 1.0),
    ("Macro-F1, single recordings", "recordings", "macro_f1", 3, 1.0),
    ("A4C read as A5C, recordings", "recordings", "a4c_as_a5c", 0, 1.0),
    ("A5C read as A4C, recordings", "recordings", "a5c_as_a4c", 0, 1.0),
    ("Label changes per recording", "recordings", "cuts_per_recording", 3, 1.0),
)


def test_part_one_of_the_ablations_other_decoders():
    from echoview.evaluation import ablations

    d = stored("ablations/stored.json")
    released, others = d["decoders"]["released"]["selection"], d["decoders"]["others"]
    compared, plain = ablations.COMPARED, ablations.NO_VIEW_TEST
    assert list(others) == [*compared, plain] and list(compared) == ["no decoder", "mode filter", "mean filter", "minimum duration", "Viterbi path"]
    minus = {name: others[name]["released_minus_this"] for name in compared}
    assert all(entry["level"] == 95.0 for apart_from in minus.values() for part in apart_from.values() for entry in part.values())
    assert {key for part in minus["no decoder"].values() for key in part} == {key for _, _, key, _, _ in COMPARED_ROWS}  # every number of the file is a row

    # the table that decides the word
    def first(s: dict) -> str:
        m, r = s["multi"], s["recordings"]
        return (f"{m['edit']['value']:.1f} | {f3(m['cut_precision']['value'])} | {f3(m['cut_recall']['value'])} "
                f"| {100 * m['same_view_cut_rate']['value']:.1f}% | "
                f"{m['extra_segments']['value']:.0f} | {m['missed_segments']['value']:.0f} | {f3(r['cuts_per_recording']['value'])}")

    def number(e: dict) -> str:
        chosen = e.get("chosen")
        if chosen is None:
            return "none"
        if isinstance(chosen["value"], dict):
            return f"half window {chosen['value']['half_window']}, threshold {chosen['value']['threshold']:.3f}"
        return f"{chosen['value']:g}"

    rows = ["| Decoder | Its number | Segment F1 at 50%, multi streams | Released minus this | 99% interval | The released decoder is | Edit | Cut precision | "
            "Cut recall | Cuts at joins of the same view | Segments that match nothing | Segments missed | Label changes per recording |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            f"| released change-point decoder | half window 7, threshold 0.174 | {f3(released['multi']['segment_f1_50']['value'])} | | | | {first(released)} |"]
    assert (d["decoders"]["released"]["setting"]["half_window"], f"{d['decoders']['released']['setting']['threshold']:.3f}") == (7, "0.174")
    for name, e in others.items():
        decides = f"[{e['decides']['interval'][0]:+.3f}, {e['decides']['interval'][1]:+.3f}]" if "decides" in e else ""
        rows.append(f"| {name} | {number(e)} | {f3(e['selection']['multi']['segment_f1_50']['value'])} "
                    f"| {e['released_minus_this']['multi']['segment_f1_50']['value']:+.3f} | {decides} | {e.get('word', '')} | {first(e['selection'])} |")
    said(ABLATIONS, "\n".join(rows))

    # the other numbers of section 5
    def second(s: dict) -> str:
        m, r = s["multi"], s["recordings"]
        return (f"{f3(s['pairs']['segment_f1_50']['value'])} | {100 * m['wrong_samples']['value']:.1f}% | {f3(m['cut_distance_mean_s']['value'])} s "
            f"| {right(s)} | "
                f"{f3(r['macro_f1']['value'])} | {whole(r['a4c_as_a5c']['value'])} | {whole(r['a5c_as_a4c']['value'])}")

    said(ABLATIONS, "\n".join(["| Decoder | Segment F1 at 50%, pair streams | Wrong samples | Mean distance of matched cuts "
                               "| Recordings right of 284, majority rule "
                               "| Macro-F1 | A4C read as A5C | A5C read as A4C |", "|---|---|---|---|---|---|---|---|",
                               f"| released change-point decoder | {second(released)} |"]
                              + [f"| {name} | {second(e['selection'])} |" for name, e in others.items()]))

    # every difference with its interval
    said(ABLATIONS, "\n".join(["| Released minus | " + " | ".join(compared) + " |", "|---|---|---|---|---|---|"]
                              + [f"| {title} | " + " | ".join(spread(minus[name][part][key], digits, scale) for name in compared) + " |"
                                 for title, part, key, digits, scale in COMPARED_ROWS]))

    intervals = sum(minus[name][part][key].get("interval") is not None for name in compared for _, part, key, _, _ in COMPARED_ROWS)
    said(ABLATIONS, f"A count of confused recordings has no interval. The {intervals} intervals are not corrected for their number. The 95% interval of every "
                    "value of the two tables above is in the file.")
    assert all("interval" in entry for e in others.values() for part in e["selection"].values() for entry in part.values())

    # the word
    words = {name: others[name]["word"] for name in compared}
    assert words == {"no decoder": "ahead of", "mode filter": "ahead of", "mean filter": "ahead of", "minimum duration": "ahead of",
                     "Viterbi path": "not shown to differ from"}
    for name in compared:
        assert others[name]["decides"]["level"] == 99.0 and words[name] == ablations.word(others[name]["decides"])
    gains = {name: minus[name]["multi"]["segment_f1_50"]["value"] for name in compared}
    assert min(gains.values()) > 0  # the released decoder has the highest value
    ahead = sorted(gains[name] for name in compared if words[name] == "ahead of")
    said(ABLATIONS, f"It is {f3(ahead[0])} to {f3(ahead[-1])} above them, and every 99% interval lies above zero.")

    # the Viterbi path
    viterbi = others["Viterbi path"]
    low, high = viterbi["decides"]["interval"]
    assert low < 0 < high
    said(ABLATIONS, f"a cost of {viterbi['chosen']['value']:g} for every change of view, reaches {f3(viterbi['selection']['multi']['segment_f1_50']['value'])} "
                    f"where the released decoder reaches {f3(released['multi']['segment_f1_50']['value'])}. The difference is "
                    f"{f3(viterbi['decides']['value'])}, and its 99% interval runs from {low:+.3f} to {high:+.3f}. This does not show that the two are equal: "
                    f"the interval allows the released decoder to be {f3(high)} better or {f3(-low)} worse.")
    shown = [f"{part}.{key}" for _, part, key, _, _ in COMPARED_ROWS if apart(minus["Viterbi path"][part][key])]
    assert shown == ["multi.cut_precision", "multi.cut_distance_mean_s"]  # "Two of the other numbers separate the two"
    precision, distance = minus["Viterbi path"]["multi"]["cut_precision"], minus["Viterbi path"]["multi"]["cut_distance_mean_s"]
    assert precision["value"] > 0 > distance["value"]  # "both for the released decoder"
    joins = whole(released["multi"]["same_view_cut_rate"]["support"])
    at_joins = [whole(joins * s["multi"]["same_view_cut_rate"]["value"]) for s in (viterbi["selection"], released)]
    changes = [whole(SINGLE * s["recordings"]["cuts_per_recording"]["value"]) for s in (viterbi["selection"], released)]
    said(ABLATIONS, f"The precision of its cuts is higher by {f3(precision['value'])}, with an interval from {precision['interval'][0]:+.3f} to "
                    f"{precision['interval'][1]:+.3f}. Its matched cuts lie nearer to the reference cuts by {f3(-distance['value'])} s, and the "
                    "interval of that "
                    f"difference in distance runs from {distance['interval'][0]:+.3f} to {distance['interval'][1]:+.3f}. The Viterbi path has fewer "
                    "cuts at joins of "
                    f"the same view ({at_joins[0]} against {at_joins[1]} of {joins} joins) and fewer label changes inside single recordings "
                    f"({changes[0]} against "
                    f"{changes[1]} over the {SINGLE}), and neither interval excludes zero.")
    assert at_joins[0] < at_joins[1] and changes[0] < changes[1]

    # single recordings, the distance of the cuts, wrong samples, pair streams
    counts = {"released": right(released), **{name: right(others[name]["selection"]) for name in compared}}
    confused = {"released": whole(released["recordings"]["a4c_as_a5c"]["value"]),
                **{name: whole(others[name]["selection"]["recordings"]["a4c_as_a5c"]["value"]) for name in compared}}
    assert counts == {"released": 281, "no decoder": 282, "mode filter": 282, "mean filter": 282, "minimum duration": 282, "Viterbi path": 281}
    assert confused == {"released": 2, "no decoder": 1, "mode filter": 1, "mean filter": 1, "minimum duration": 1, "Viterbi path": 2}
    accuracy = minus["no decoder"]["recordings"]["accuracy"]
    assert all(minus[name]["recordings"]["accuracy"] == accuracy for name in compared[:4])
    said(ABLATIONS, f"By the majority rule the released decoder and the Viterbi path read {counts['released']} of the {SINGLE} recordings right, "
        "with A4C read as "
                    f"A5C twice. No decoder, the two filters and the minimum duration read {counts['no decoder']}, with A4C read as A5C once. The "
                    "difference is one "
                    f"recording, {spread(accuracy)} in accuracy.")
    seconds = {"released": released["multi"]["cut_distance_mean_s"]["value"],
               **{name: others[name]["selection"]["multi"]["cut_distance_mean_s"]["value"] for name in compared}}
    assert [name for name in compared if seconds[name] < seconds["released"]] == ["mean filter"]
    shown_apart = [name for name in compared if apart(minus[name]["multi"]["cut_distance_mean_s"])]
    assert shown_apart == ["mode filter", "mean filter", "minimum duration", "Viterbi path"]
    nearer = minus["mean filter"]["multi"]["cut_distance_mean_s"]
    said(ABLATIONS, f"They lie {f3(seconds['released'])} s from the reference cuts on average. Those of the mean filter lie at {f3(seconds['mean filter'])} s, "
                    f"nearer by {f3(nearer['value'])} s ({spread(nearer)}). Those of the mode filter, the minimum duration and the Viterbi path lie "
                    "farther, at "
                    f"{f3(seconds['mode filter'])} s, {f3(seconds['minimum duration'])} s and {f3(seconds['Viterbi path'])} s, and each of these "
                    "intervals excludes "
                    f"zero. Without a decoder they lie at {f3(seconds['no decoder'])} s, and that interval includes zero.")
    wrong = {"released": released["multi"]["wrong_samples"]["value"],
             **{name: others[name]["selection"]["multi"]["wrong_samples"]["value"] for name in compared}}
    assert min(wrong, key=wrong.get) == "released"
    assert [name for name in compared if apart(minus[name]["multi"]["wrong_samples"])] == ["no decoder", "mode filter", "minimum duration"]
    rest = sorted(wrong[name] for name in compared if name != "Viterbi path")
    said(ABLATIONS, f"**The released decoder has the fewest wrong samples,** {100 * wrong['released']:.1f}%, against "
        f"{100 * wrong['Viterbi path']:.1f}% for the "
                    f"Viterbi path and {100 * rest[0]:.1f}% to {100 * rest[-1]:.1f}% for the others. The interval of the difference excludes zero "
                    "against no decoder, "
                    "the mode filter and the minimum duration.")
    pairs = {"released": released["pairs"]["segment_f1_50"]["value"],
             **{name: others[name]["selection"]["pairs"]["segment_f1_50"]["value"] for name in compared}}
    assert pairs["mode filter"] == pairs["minimum duration"] == pairs["released"]
    assert not any(minus[name]["pairs"]["segment_f1_50"]["interval"][0] > 0 for name in compared)
    said(ABLATIONS, f"**On pair streams the mode filter and the minimum duration equal the released decoder,** at {f3(pairs['released'])}. The "
        "Viterbi path reaches "
                    f"{f3(pairs['Viterbi path'])}, no decoder {f3(pairs['no decoder'])} and the mean filter {f3(pairs['mean filter'])}. No interval "
                    "lies wholly "
                    "above zero.")

    # without the view test, and how the numbers were chosen
    published_setting = next(s for s in d["decoder_settings"]["settings"] if (s["half_window"], s["threshold_quantile"]) == (7, 0.999))
    assert others[plain]["selection"] == released and "word" not in others[plain]  # every number the same
    assert all(entry["value"] == 0.0 and entry["interval"] in ([0.0, 0.0], None)
               for part in others[plain]["released_minus_this"].values() for entry in part.values())
    said(ABLATIONS, "**Without the view test no number changes.** Without the test the rule chooses the same setting, every number of the tables comes out the "
                    "same, and every interval of a difference is [0, 0]. This says little about the test.",
         "So the numbers stay the same when the test drops no peak, and they stay the same when the pieces on the two sides of every dropped peak "
         "take the same view. They do not say which of the two it is. The file does not hold how many peaks the test drops.")
    assert others[plain]["chosen"]["value"] == {key: published_setting[key] for key in others[plain]["chosen"]["value"]}  # "the rule chooses the same setting"
    assert "peaks" not in json.dumps(d)  # the file holds no count of peaks
    feasible = {name: (others[name]["chosen"]["n_feasible"], others[name]["chosen"]["grid"], others[name]["chosen"]["floors_dropped"])
                for name in list(others)[1:]}
    assert feasible == {"mode filter": (6, 8, False), "mean filter": (6, 8, False), "minimum duration": (4, 7, False), "Viterbi path": (6, 8, False),
                        plain: (40, 40, False)}
    assert others["mean filter"]["chosen"]["value"] == min(ablations.GRIDS["mean filter"]) == 3
    on_calibration = {name: others[name]["chosen"]["calibration"]["segment_f1_50_multi"] for name in compared[1:]}
    published = next(s for s in d["decoder_settings"]["settings"] if (s["half_window"], s["threshold_quantile"]) == (7, 0.999))
    said(ABLATIONS, "6 of 8 for the mode filter, the mean filter and the Viterbi path, and 4 of 7 for the minimum duration. No floor was dropped. "
                    "The window of the mean filter, 3, is the smallest that was tried.",
         f"the released decoder {f3(published['calibration']['segment_f1_50_multi'])}, the mode filter {f3(on_calibration['mode filter'])}, the "
         "minimum duration "
         f"{f3(on_calibration['minimum duration'])}, the Viterbi path {f3(on_calibration['Viterbi path'])} and the mean filter "
         f"{f3(on_calibration['mean filter'])}.",
         f"The two filters did not: they are {f3(gains['mode filter'])} and {f3(gains['mean filter'])} below.")
    assert gains["Viterbi path"] < 0.03 < min(gains["mode filter"], gains["mean filter"])  # "within 0.03 ... as expected. The two filters did not"
    here = [published["calibration"]["segment_f1_50_multi"], *on_calibration.values()]
    there = [released["multi"]["segment_f1_50"]["value"], *(others[name]["selection"]["multi"]["segment_f1_50"]["value"] for name in compared[1:])]
    assert max(here) - min(here) < 0.01 < max(there) - min(there)  # "the decoders lie closer together than on the selection half"
    said(ABLATIONS, "**Two of the other numbers separate the two, both for the released decoder.**",
         "**On single recordings the released decoder is not the best.**", "**Its matched cuts are not the nearest.**",
         "**It is not shown to differ from a Viterbi path.**",
         "**The released decoder is ahead of no decoder, of both filters and of the minimum duration.**",
         "On the calibration half, where the numbers were chosen, the decoders lie closer together than on the selection half")
    assert others["no decoder"]["word"] == "ahead of"  # "doing without a decoder is clearly behind"


def test_part_one_of_the_ablations_the_settings_of_the_decoder():
    from echoview.evaluation import calibrate

    d = stored("ablations/stored.json")
    settings, others = d["decoder_settings"], d["decoders"]["others"]
    every = settings["settings"]
    assert len(every) == 40 and sorted({s["threshold_quantile"] for s in every}) == [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
    grid: dict[int, list[str]] = {}
    for s in every:
        grid.setdefault(s["half_window"], []).append(f3(s["selection"]["segment_f1_50_multi"]))
    said(ABLATIONS, "\n".join(["| Half window | 0.80 | 0.85 | 0.90 | 0.95 | 0.97 | 0.99 | 0.995 | 0.999 |", "|---|---|---|---|---|---|---|---|---|"]
                              + [f"| {half_window} | " + " | ".join(values) + " |" for half_window, values in grid.items()]))
    columns = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]  # every row of the table runs over the same quantiles, in this order
    assert all([s["threshold_quantile"] for s in every if s["half_window"] == half_window] == columns for half_window in grid)

    def three(r: dict) -> str:
        return f"{f3(r['segment_f1_50_multi'])} | {f3(r['native_macro_f1'])} | {f3(r['boundary_recall_multi'])}"

    said(ABLATIONS, "\n".join(["| Half window | Quantile | Threshold | Calibration half: segment F1 | Macro-F1 | Boundary recall "
                               "| Selection half: segment F1 | "
                               "Macro-F1 | Boundary recall |", "|---|---|---|---|---|---|---|---|---|"]
                              + [f"| {s['half_window']} | {s['threshold_quantile']:g} | {s['threshold']:.5f} | {three(s['calibration'])} | "
                                  f"{three(s['selection'])} |"
                                 for s in every]))

    summary = settings["selection_segment_f1_50_multi"]
    counted = (summary["within_0.01_of_the_published"], summary["within_0.02_of_the_published"], summary["above_the_published"], summary["settings"])
    assert counted == (3, 7, 0, 40)
    best = max(every, key=lambda s: s["selection"]["segment_f1_50_multi"])
    published = next(s for s in every if (s["half_window"], s["threshold_quantile"]) == (7, 0.999))
    assert best is published and published["threshold_quantile"] == max(s["threshold_quantile"] for s in every)  # "the highest on the grid"
    said(ABLATIONS, "Three of the 40 settings lie within 0.01 of that value, the published one among them, and seven within 0.02. None lies above it.",
         f"The lowest is {f3(summary['lowest'])} and the median {f3(summary['median'])}.", "**The published threshold is the highest on the grid.**")

    # the five lower quantiles, and the two above them
    mode, duration = (others[name]["selection"]["multi"]["segment_f1_50"]["value"] for name in ("mode filter", "minimum duration"))
    lower = [s["selection"]["segment_f1_50_multi"] for s in every if s["threshold_quantile"] <= 0.97]
    middle = [s["selection"]["segment_f1_50_multi"] for s in every if s["threshold_quantile"] in (0.99, 0.995)]
    above = sorted(value for value in middle if value > max(mode, duration))
    assert len(lower) == 25 and max(lower) < duration < mode and (len(middle), len(above)) == (10, 4)
    said(ABLATIONS, "**At the five lower quantiles the decoder is below the mode filter and the minimum duration.** "
                    f"There every half window stays between {f3(min(lower))} and {f3(max(lower))}, where the minimum duration reaches {f3(duration)} "
                    "and the mode "
                    f"filter {f3(mode)}. At the 0.99 and 0.995 quantiles four of the ten settings are above both, with {f3(above[0])} to {f3(above[-1])}.")

    mean = others["mean filter"]["selection"]["multi"]["segment_f1_50"]["value"]
    said(ABLATIONS, f"At the five lower quantiles the decoder is below those two, and {sum(value > mean for value in lower)} of its 25 settings there "
                    "are above the mean filter.",
         "**The decoder depends on its threshold.** The published setting, half window 7 at the 0.999 quantile, has the highest value of the 40.")

    # the floors on both halves
    macro, recall = calibrate.NATIVE_FLOOR, calibrate.RECALL_FLOOR
    assert (macro, recall) == (0.980, 0.95)
    assert settings["feasible_on_calibration"]["settings"] == 40 and all(s["feasible_on_calibration"] for s in every)
    assert all(s["calibration"]["native_macro_f1"] >= macro and s["calibration"]["boundary_recall_multi"] >= recall for s in every)
    low = [s for s in every if s["selection"]["native_macro_f1"] < macro]
    assert len(low) == 7 and any(s is published for s in low) and all(s["selection"]["boundary_recall_multi"] >= recall for s in every)
    near = [s for s in every if s is not published and abs(s["selection"]["segment_f1_50_multi"] - published["selection"]["segment_f1_50_multi"]) <= 0.02]
    assert len(near) == 6 and sum(any(s is other for other in low) for s in near) == 5
    said(ABLATIONS, "**All 40 settings keep the two floors on the calibration half,** so there the rule decides by segment F1 alone. On the "
        "selection half 7 of the 40 "
                    f"fall below the floor of 0.980 for the macro-F1 of single recordings. The published setting is one of them, with "
                    f"{f3(published['selection']['native_macro_f1'])}, and so are five of the six other settings within 0.02 of it. None falls below "
                    "the floor of 0.95 "
                    "for the boundary recall.")

    # the halves exchanged
    exchanged = settings["halves_exchanged"]
    taken = (exchanged["setting"]["half_window"], exchanged["setting"]["threshold_quantile"])
    cell = next(s for s in every if (s["half_window"], s["threshold_quantile"]) == taken)
    assert abs(cell["threshold"] - exchanged["setting"]["threshold"]) > 0.01  # "This is not a cell of the table above"
    assert set(exchanged) == {"status", "setting", "calibration"} and set(exchanged["calibration"]) == {"chosen_on_selection", "published"}  # no more is held
    said(ABLATIONS, f"the rule takes half window {exchanged['setting']['half_window']} with a threshold of {f3(exchanged['setting']['threshold'])}, the "
                    f"{exchanged['setting']['threshold_quantile']:g} quantile of the scores of the selection half. This is not a cell of the table "
                    "above, whose "
                    f"thresholds come from the calibration half: its cell for half window {cell['half_window']} and {cell['threshold_quantile']:g} "
                    "has a threshold of "
                    f"{f3(cell['threshold'])}. On the calibration half the exchanged choice gives "
                    f"{f3(exchanged['calibration']['chosen_on_selection']['segment_f1_50_multi'])}, where the published setting gives "
                    f"{f3(exchanged['calibration']['published']['segment_f1_50_multi'])}. The file does not hold what the exchanged choice reaches "
                    "on the selection "
                    "half, nor how many settings kept the floors there.")
    assert exchanged["calibration"]["published"] == published["calibration"]


def test_part_one_of_the_ablations_one_temperature_and_what_the_changelog_says():
    from echoview.evaluation import ablations

    d = stored("ablations/stored.json")
    temperature = d["temperature"]
    curve = {entry["temperature"]: entry["log_loss"] for entry in temperature["calibration_curve"]}
    assert temperature["chosen"] == min(curve) == min(ablations.TEMPERATURES) == 0.3 and temperature["word"] == "candidate"
    said(ABLATIONS, f"The fit on the calibration half chose {temperature['chosen']:.2f}, the lowest value that was tried.",
         f"falls from {f3(curve[1.0])} at a temperature of 1 to {f3(curve[0.5])} at 0.5, and stays near {f3(curve[0.3])} below that. "
         "The intervals of this part draw recordings.")
    assert f3(curve[0.4]) == f3(curve[0.3]) == "0.015"
    rows = ["| The 284 single recordings of the selection half | Wrong | Mean probability | Right | Mean probability minus the share that is right | "
            "Calibration error, equal width | Calibration error, equal count | Brier score | Log loss |", "|---|---|---|---|---|---|---|---|---|"]
    for name, e in temperature["selection"].items():
        gap = e["probability_minus_right"]
        rows.append(f"| {name} | {e['errors']} | {f3(e['mean_probability'])} | {round(SINGLE * e['share_right'])} | {gap['value']:+.3f} "
                    f"[{gap['interval'][0]:+.3f}, {gap['interval'][1]:+.3f}] | {f3(e['ece'])} | {f3(e['ece_equal_count'])} | {f3(e['brier']['value'])} "
                    f"| {f3(e['log_loss']['value'])} |")
    said(ABLATIONS, "\n".join(rows))
    loss, brier = temperature["with_minus_published"]["log_loss"], temperature["with_minus_published"]["brier"]
    before, after = (temperature["selection"][name] for name in ("as published", "with the temperature"))
    assert loss["level"] == brier["level"] == 95.0 and loss["interval"][1] < 0 and after["errors"] <= before["errors"]  # the plan's rule for "candidate"
    said(ABLATIONS, f"**The temperature is a candidate.** The log loss with the temperature minus the log loss as published is {spread(loss)}, the "
        "same difference "
                    f"for the Brier score is {spread(brier)}, and no more recordings are wrong.",
         f"less sure than it is right by {f3(-before['probability_minus_right']['value'])} as published, and by "
         f"{f3(-after['probability_minus_right']['value'])} with the temperature")

    # what the document says of the test benchmark comes from the read for `classify`
    found = stored("evaluation/classify_test.json")["classify"]
    top = [b for b in found["probability"]["bins"] if b["above"] >= 0.9 - 1e-9]
    assert (found["recordings"], found["errors"]) == (800, 14) and sum(b["recordings"] - b["right"] for b in top) == 2
    said(ABLATIONS, "On the test benchmark 14 of 800 recordings are wrong by `classify`'s rule, two of them with a probability above 0.9")

    # the changelog
    systems, others = d["ensemble_size"]["systems"], d["decoders"]["others"]
    released = d["decoders"]["released"]["selection"]
    f1 = {name: e["selection"]["multi"]["segment_f1_50"]["value"] for name, e in systems.items()}
    singles = [name for name in systems if name.startswith("seed ")]
    best = max(singles, key=f1.get)
    summary = d["decoder_settings"]["selection_segment_f1_50_multi"]
    more = {right(others[name]["selection"]) for name in ("no decoder", "mode filter", "mean filter", "minimum duration")}
    assert more == {282} and right(released) == 281 and summary["within_0.02_of_the_published"] == 7 and summary["settings"] == 40
    said("CHANGELOG.md",
         f"On the multi streams the three seeds together have the highest segment F1 ({f3(f1[WHOLE])}), and the best single seed is "
         f"{f3(-systems[best]['minus_the_whole_ensemble']['multi']['segment_f1_50']['value'])} below them. On pair streams, and for single recordings by "
         "the majority rule, they are not the highest.",
         "For segment F1 on the multi streams the released decoder is ahead of no decoder, of a mode filter, of a mean filter and of a minimum duration",
         "a Viterbi path with one number is not shown to differ from the released decoder "
         f"({f3(others['Viterbi path']['selection']['multi']['segment_f1_50']['value'])} "
         f"against {f3(released['multi']['segment_f1_50']['value'])})",
         "By the majority rule no decoder, the two filters and the minimum duration read one single recording more right than the released decoder "
         f"(282 against 281 of {SINGLE}).",
         "7 of its 40 settings, the published one among them, lie within 0.02 of the published one",
         f"One temperature of {temperature['chosen']:.2f} lowers the log loss of the single recordings and is a candidate, not a change.",
         "No number was wrong.")


# ---------------------------------------------------------------------------------------------------------------------
# Parts 2 and 3 of docs/ABLATIONS.md. Their results are written by tools/write_ablation_results.py from the two output
# files, and the tool asserts what the prose says without a number. The document must hold what the tool prints, to
# the letter. The second test reaches a few headline numbers without the tool, and the sentences of the other documents.

def writer_of_the_results():
    import importlib.util

    spec = importlib.util.spec_from_file_location("write_ablation_results", ROOT / "tools" / "write_ablation_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parts_two_and_three_of_the_ablations_are_what_the_tool_writes_from_their_files():
    import copy

    writer = writer_of_the_results()
    forward, left_open, part_one = stored("ablations/forward.json"), stored("ablations/open.json"), stored("ablations/stored.json")
    said(ABLATIONS, writer.part_two(forward), writer.part_three(left_open, part_one))
    # a file that says something else cannot be written up: the tool's own checks stop it
    other = copy.deepcopy(forward)
    other["systems"]["published"]["conditions"]["mirrored"]["minus_unchanged"]["multi"]["segment_f1_50"]["value"] = -0.01  # no longer the largest loss
    with pytest.raises(AssertionError):
        writer.part_two(other)
    other = copy.deepcopy(forward)
    other["published_minus_stage2"]["mirrored"]["multi"]["segment_f1_50"]["interval"] = [-0.044, -0.001]  # stage 2 would be ahead under one condition
    with pytest.raises(AssertionError):
        writer.part_two(other)
    other = copy.deepcopy(left_open)
    other["higher_thresholds"]["word"] = "candidate"
    with pytest.raises(AssertionError):
        writer.part_three(other, part_one)
    other = copy.deepcopy(left_open)
    other["view_test"]["selection"]["multi"]["samples_that_change"] = 3
    with pytest.raises(AssertionError):
        writer.part_three(other, part_one)
    other = copy.deepcopy(left_open)
    other["temperature"]["calibration_curve"][5]["log_loss"] += 1e-9  # the value of part 1 at 0.30 must come out again
    with pytest.raises(AssertionError):
        writer.part_three(other, part_one)


def test_the_headline_numbers_of_parts_two_and_three_stand_in_the_documents():
    import re

    forward, left_open, part_one = stored("ablations/forward.json"), stored("ablations/open.json"), stored("ablations/stored.json")
    run = (forward["benchmark"], forward["half"], forward["sequences"], forward["draws"], forward["seed"])
    assert run == ("development", "selection", 541, 10_000, 20261123)
    assert (left_open["benchmark"], left_open["draws"], left_open["seed"]) == ("development", 10_000, 20261123)
    assert forward["sha256"]["logits"] == left_open["sha256"]["logits"] == part_one["sha256"]["logits"]  # one file of stored logits under all three
    assert forward["unchanged_against_stored"] < 1e-6 and left_open["published_selection_f1"] == part_one["published_selection_f1"]
    for name in ("forward.json", "open.json"):  # pooled numbers only: no key of a recording, nothing that is not a number
        raw = (DOCS / "ablations" / name).read_text()
        assert not re.search(r"(?<![0-9a-f.])(?=\d*[a-f])[0-9a-f]{16}(?![0-9a-f])", raw) and not re.search(r"NaN|Infinity", raw)

    # part 2, without the tool
    published, stage2 = (forward["systems"][name]["conditions"] for name in ("published", "stage 2"))

    def f1(conditions: dict, name: str) -> float:
        return conditions[name]["selection"]["multi"]["segment_f1_50"]["value"]

    def lost(conditions: dict, name: str) -> dict:
        return conditions[name]["minus_unchanged"]["multi"]["segment_f1_50"]

    assert f"{f1(published, 'unchanged'):.4f}" == "0.9594" and len(published) == 27
    said(ABLATIONS, f"| mirrored | {f3(f1(published, 'mirrored'))} | {spread(lost(published, 'mirrored'))} |",
         f"| still frame | {f3(f1(published, 'still frame'))} | {spread(lost(published, 'still frame'))} |",
         f"| unchanged | {f3(f1(stage2, 'unchanged'))} |  |  |",
         f"| noise 25 | {spread(forward['published_minus_stage2']['noise 25']['multi']['segment_f1_50'])} |",
         f"| unchanged | {spread(forward['published_minus_stage2']['unchanged']['multi']['segment_f1_50'])} |")
    losses = {name: -lost(published, name)["value"] for name in published if name != "unchanged"}
    ranked = sorted(losses, key=losses.get, reverse=True)
    assert ranked[:3] == ["mirrored", "still frame", "half the frame rate"]
    ninth = forward["systems"]["published"]["ninth_view"]
    named = [(ninth[name]["recordings"], ninth[name]["given"]["PSAX"]) for name in ("validation", "training")]
    assert named == [(46, 46), (319, 318)]
    chance = ninth["training"]["known_above_ninth"]["probability"]
    said(ABLATIONS, f"for the larger training list {f3(chance['value'])} [{f3(chance['interval'][0])}, {f3(chance['interval'][1])}]")
    assert chance["interval"][0] < 0.5 < chance["interval"][1]

    # the two limits in the model card and in the README
    errors = {name: published[name]["classify"]["errors"] for name in published}
    limit = ("- **Mirrored images, and windows that show one frame.** On development data, every frame mirrored left to right lowers segment F1 on the "
             f"multi streams from {f3(f1(published, 'unchanged'))} to {f3(f1(published, 'mirrored'))}, and `classify` then reads {errors['mirrored']} of 284 "
             f"single recordings wrong where it reads {errors['unchanged']} wrong. No training stage mirrored a frame; whether that is the cause was not "
             f"tested. When every window shows one frame sixteen times, segment F1 falls to {f3(f1(published, 'still frame'))} and `classify` reads "
             f"{errors['still frame']} wrong. None of the 24 other changes of grey levels, image quality, geometry and timing that were tried lowers segment "
             f"F1 by more than {f3(losses[ranked[2]])}. For single recordings the hardest of them are added noise with a spread of 25 grey levels and added "
             f"speckle of 0.25, with {errors['noise 25']} and {errors['speckle 0.25']} read wrong")
    said("MODEL_CARD.md", limit)
    said("README.md", limit)
    assert len(ranked) - 2 == 24
    hardest = sorted((name for name in ranked[2:]), key=lambda name: -errors[name])[:2]
    assert hardest == ["noise 25", "speckle 0.25"] and errors["speckle 0.25"] > max(errors[name] for name in ranked[2:] if name not in hardest)
    from echoview.training import augment

    assert not any(word in Path(augment.__file__).read_text().lower() for word in ("flip", "mirror"))  # "No training stage mirrored a frame"

    # part 3, without the tool
    higher = left_open["higher_thresholds"]
    chosen = higher["settings"][higher["rule"]["chosen"]]
    assert (chosen["half_window"], chosen["threshold_quantile"], higher["word"]) == (7, 0.999, "the published setting stays")
    assert all(s["selection"]["segment_f1_50_multi"] < chosen["selection"]["segment_f1_50_multi"] for s in higher["settings"] if s["added"])
    said(ABLATIONS, "**The published setting stays.** Over the 55 settings the rule takes half window 7 at the 0.999 quantile again.")
    count = left_open["view_test"]["selection"]["multi"]
    said(ABLATIONS, f"| selection | multi streams | {count['streams']} | {count['peaks']} | {count['dropped']} | {count['streams_with_a_dropped_peak']} | "
                    f"{count['streams_that_change']} | {count['samples_that_change']} |")
    temperature = left_open["temperature"]
    assert temperature["chosen"] == 0.05 and temperature["is_the_lowest_tried"] is True
    kept = temperature["streams"]["the temperature, the published setting"]["minus_published"]["multi"]["segment_f1_50"]
    said(ABLATIONS, f"| the temperature, the published setting | half window 7, threshold 0.174 | "
                    f"{f3(temperature['streams']['the temperature, the published setting']['selection']['multi']['segment_f1_50']['value'])} | "
                    f"{spread(kept)} |")

    # the changelog
    peaks = sum(kind["peaks"] for kind in left_open["view_test"]["selection"].values())
    dropped = sum(kind["dropped"] for kind in left_open["view_test"]["selection"].values())
    difference = forward["published_minus_stage2"]
    said("CHANGELOG.md",
         f"A mirrored image is the largest loss on the multi streams: segment F1 falls from {f3(f1(published, 'unchanged'))} to "
         f"{f3(f1(published, 'mirrored'))}, and the stage-2 ensemble loses {f3(-lost(stage2, 'mirrored')['value'])} under it.",
         f"When every window shows one frame sixteen times the loss is {f3(losses['still frame'])}, and none of the 24 other conditions costs more than "
         f"{f3(losses[ranked[2]])}; shuffled frames cost {f3(losses['shuffled frames'])}.",
         f"the stage-2 ensemble has {f3(f1(stage2, 'unchanged'))} where the published system has {f3(f1(published, 'unchanged'))}, "
         "a difference that is not shown",
         f"by {f3(difference['noise 25']['multi']['segment_f1_50']['value'])} under noise with a spread of 25 grey levels, by "
         f"{f3(difference['blur 2 pixels']['multi']['segment_f1_50']['value'])} under a blur of 2 pixels and by "
         f"{f3(difference['64 pixels']['multi']['segment_f1_50']['value'])} at 64 pixels",
         f"The decoder's view test drops {dropped} of {peaks} peaks on the selection half and changes no sample there.",
         "Two of the Viterbi path's 8 costs, the chosen one and the next, lie within 0.02 of the chosen one.",
         f"The fit of the temperature takes {temperature['chosen']:.2f}, the lowest value tried again, and at that temperature the multi streams lose "
         f"{f3(-kept['value'])}.")
    viterbi = left_open["other_decoders"]["Viterbi path"]
    near = [row["value"] for row in viterbi["values"]
            if abs(row["selection"]["segment_f1_50_multi"] - viterbi["selection_segment_f1_50_multi"]["of_the_chosen"]) <= 0.02]
    assert near == [8.0, 16.0] and viterbi["chosen"] == 8.0 and viterbi["selection_segment_f1_50_multi"]["within_0.02_of_the_chosen"] == 2
    assert sum(kind["samples_that_change"] for kind in left_open["view_test"]["selection"].values()) == 0  # "changes no sample there"
    interval = difference["unchanged"]["multi"]["segment_f1_50"]["interval"]
    assert interval[0] < 0 < interval[1]  # "a difference that is not shown"
    largest = sorted((difference[name]["multi"]["segment_f1_50"]["value"], name) for name in difference)[-5:]
    assert {name for _, name in largest} == {"noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5"}  # the "five conditions"
