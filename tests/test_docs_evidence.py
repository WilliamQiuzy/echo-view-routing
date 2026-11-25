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
    said("docs/EVALUATION.md", "Twelve of the fifteen errors are between A4C and A5C", "| Test benchmark | 800 | 15 | not measured |")
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
