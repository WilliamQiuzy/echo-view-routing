"""The results of parts 2 and 3 of docs/ABLATIONS.md, written from their output files.

  python tools/write_ablation_results.py part2 docs/ablations/forward.json
  python tools/write_ablation_results.py part3 docs/ablations/open.json docs/ablations/stored.json

Each command prints a section of the document. Every number of the text comes from the file. Many sentences that say
something without a number (a largest loss, an interval on one side of zero, a count) are asked of the file first,
and nothing is printed when the file says otherwise. Not every sentence can be asked of a file, so the text was also
read against the files by an independent reader. tests/test_docs_evidence.py holds the document to what this prints.
The checks are assertions, so the tool does not run under `python -O`.
"""
from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path


def part_two(d: dict) -> str:
    """Section 9, part 2: the published system and the stage-2 ensemble under every condition, and the ninth view."""
    PUBLISHED, STAGE2, CLEAN = "published", "stage 2", "unchanged"
    P, S = d["systems"][PUBLISHED], d["systems"][STAGE2]
    NAMES = list(P["conditions"])
    CHANGED = [name for name in NAMES if name != CLEAN]
    RECORDINGS = 284
    VIEWS = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
    NUMBERS = (  # the column's name, the part, the key, the digits, the scale: every number of section 5
        ("Segment F1, multi", "multi", "segment_f1_50", 3, 1.0),
        ("Edit", "multi", "edit", 1, 1.0),
        ("Wrong samples, points", "multi", "wrong_samples", 1, 100.0),
        ("Cut precision", "multi", "cut_precision", 3, 1.0),
        ("Cut recall", "multi", "cut_recall", 3, 1.0),
        ("Cut distance, s", "multi", "cut_distance_mean_s", 3, 1.0),
        ("Cuts at same-view joins, points", "multi", "same_view_cut_rate", 1, 100.0),
        ("Segments that match nothing", "multi", "extra_segments", 0, 1.0),
        ("Segments missed", "multi", "missed_segments", 0, 1.0),
        ("Segment F1, pairs", "pairs", "segment_f1_50", 3, 1.0),
        ("Accuracy", "recordings", "accuracy", 3, 1.0),
        ("Macro-F1", "recordings", "macro_f1", 3, 1.0),
        ("A4C as A5C", "recordings", "a4c_as_a5c", 0, 1.0),
        ("A5C as A4C", "recordings", "a5c_as_a4c", 0, 1.0),
        ("Label changes per recording", "recordings", "cuts_per_recording", 3, 1.0),
    )
    LOWER_IS_BETTER = ("wrong_samples", "cut_distance_mean_s", "same_view_cut_rate", "extra_segments", "missed_segments", "a4c_as_a5c", "a5c_as_a4c",
                       "cuts_per_recording")


    ties = []  # the numbers that are a tie of the last digit shown only because they are taken at twelve decimals first


    def rounded(x, digits, sign=False, counted=True):
        """A number at the digits shown, with a tie rounded away from zero as in ordinary rounding (Python's own formatting
        rounds a tie of the binary number to even). The number is taken at twelve decimals first, so that a tie which the
        file stores a hair off itself is the tie."""
        step = Decimal(1).scaleb(-digits)
        number = Decimal(repr(float(x)))
        value = number.quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN).quantize(step, rounding=ROUND_HALF_UP)
        if counted and value != number.quantize(step, rounding=ROUND_HALF_UP):
            ties.append((float(x), digits))
        return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


    def f3(x):
        return rounded(x, 3)


    def signed(entry, digits=3, scale=1.0):
        if entry.get("interval") is None:
            return rounded(scale * entry["value"], digits, True)
        low, high = entry["interval"]
        ends = f"[{rounded(scale * low, digits, True)}, {rounded(scale * high, digits, True)}]"
        return f"{rounded(scale * entry['value'], digits, True)} {ends}"


    def below(entry):
        return entry.get("interval") is not None and entry["interval"][1] < 0


    def above(entry):
        return entry.get("interval") is not None and entry["interval"][0] > 0


    def at_zero(entry):
        """An interval that excludes zero and whose nearer end is zero to three decimals."""
        low, high = entry["interval"]
        return (0 < low < 0.0005) or (-0.0005 < high < 0)


    def whole(x):
        assert abs(x - round(x)) < 1e-6, x
        return round(x)


    def value(system, name, part="multi", key="segment_f1_50"):
        return system["conditions"][name]["selection"][part][key]["value"]


    def loss(system, name, part="multi", key="segment_f1_50"):
        return system["conditions"][name]["minus_unchanged"][part][key]


    def right(system, name):
        return whole(RECORDINGS * value(system, name, "recordings", "accuracy"))


    def changes(system, name):
        return whole(RECORDINGS * value(system, name, "recordings", "cuts_per_recording"))


    def listed(names):
        names = list(names)
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


    # ------------------------------------------------------------------------------------------------- the tables
    def system_rows(system):
        rows = ["| Condition | Segment F1 at 50%, multi streams | Minus the unchanged frames | Pair streams, minus unchanged | "
                "Recordings right of 284, majority rule | Wrong by `classify`'s rule | Mean probability | Recordings in which the seeds differ |",
                "|---|---|---|---|---|---|---|---|"]
        for name in NAMES:
            c = system["conditions"][name]
            multi = "" if name == CLEAN else signed(loss(system, name))
            pairs = "" if name == CLEAN else signed(loss(system, name, "pairs"))
            rows.append(f"| {name} | {f3(value(system, name))} | {multi} | {pairs} | {right(system, name)} | {c['classify']['errors']} | "
                        f"{f3(c['classify']['mean_probability'])} | {c['classify']['seeds_differ']} |")
        return rows


    def difference_rows():
        apart = d["published_minus_stage2"]
        rows = ["| Condition | Segment F1 at 50%, multi streams | Pair streams | Accuracy, single recordings | Wrong samples, percentage points |",
                "|---|---|---|---|---|"]
        for name in NAMES:
            a = apart[name]
            rows.append(f"| {name} | {signed(a['multi']['segment_f1_50'])} | {signed(a['pairs']['segment_f1_50'])} | "
                        f"{signed(a['recordings']['accuracy'])} | {signed(a['multi']['wrong_samples'], 1, 100.0)} |")
        return rows


    def every_number_rows(entries, names):
        """One row per condition and one column per number of section 5, each a difference with its interval."""
        rows = ["| Condition | " + " | ".join(title for title, _, _, _, _ in NUMBERS) + " |", "|---|" + "---|" * len(NUMBERS)]
        for name in names:
            cells = [signed(entries[name][part][key], digits, scale) for _, part, key, digits, scale in NUMBERS]
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        return rows


    def ninth_rows():
        rows = ["| System | List | Recordings | Named PLAX | PSAX | A4C | A5C | SC4C | Probability: lowest | quarter | median | three quarters | highest | "
                "Above 0.9 | Seeds differ |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for title, system in (("published", P), ("stage-2 ensemble", S)):
            for label in ("validation", "training"):
                n = system["ninth_view"][label]
                p = n["probability"]
                named = " | ".join(str(n["given"][view]) for view in VIEWS)
                rows.append(f"| {title} | {label} | {n['recordings']} | {named} | {f3(p['lowest'])} | {f3(p['quarter'])} | {f3(p['median'])} | "
                            f"{f3(p['three_quarters'])} | {f3(p['highest'])} | {p['above_0.9']} | {n['seeds_differ']} |")
        return rows


    def apart_rows():
        rows = ["| System | List | The largest probability | Its margin over the second | The negative entropy |", "|---|---|---|---|---|"]
        for title, system in (("published", P), ("stage-2 ensemble", S)):
            for label in ("validation", "training"):
                k = system["ninth_view"][label]["known_above_ninth"]
                cells = [f"{f3(k[key]['value'])} [{f3(k[key]['interval'][0])}, {f3(k[key]['interval'][1])}]"
                         for key in ("probability", "margin", "negative_entropy")]
                rows.append(f"| {title} | {label} | " + " | ".join(cells) + " |")
        return rows


    # --------------------------------------------------------------------------------------------------- the text
    def text():
        assert (d["benchmark"], d["half"], d["sequences"], d["draws"], d["seed"]) == ("development", "selection", 541, 10_000, 20261123)
        assert len(NAMES) == 27 and NAMES[0] == CLEAN and d["unchanged_against_stored"] < 1e-6
        assert f"{rounded(value(P, CLEAN), 4)}" == "0.9594"
        kinds = {name: P["conditions"][name]["kind"] for name in NAMES}
        image = [name for name in CHANGED if kinds[name] == "image"]
        time = [name for name in CHANGED if kinds[name] == "time"]
        assert len(image) == 22 and time == ["still frame", "shuffled frames", "played backwards", "half the frame rate"]

        # the published system: where it loses
        losses = {name: loss(P, name) for name in CHANGED}
        lost = [name for name in CHANGED if below(losses[name])]
        assert [name for name in CHANGED if above(losses[name])] == []  # no interval lies above zero
        held = [name for name in CHANGED if not below(losses[name])]
        worst = sorted(CHANGED, key=lambda name: losses[name]["value"])
        assert worst[:2] == ["mirrored", "still frame"]
        third = losses[worst[2]]["value"]
        mirrored, still = losses["mirrored"], losses["still frame"]
        higher = sorted(losses[name]["value"] for name in CHANGED if losses[name]["value"] > 0)
        grey = ["gamma 0.5", "gamma 0.7", "gamma 1.5", "gamma 2.0", "darker by 40", "brighter by 40", "contrast 0.5", "contrast 1.5"]
        degrade = ["noise 10", "noise 25", "speckle 0.25", "blur 1 pixel", "blur 2 pixels", "JPEG quality 30", "JPEG quality 10", "112 pixels", "64 pixels"]
        geometry = ["mirrored", "zoomed in 1.2", "zoomed out 0.8", "rotated 10 degrees", "rotated 20 degrees"]
        assert grey + degrade + geometry == image
        grey_lost = [name for name in grey if name in lost]
        degrade_held = [name for name in degrade if name in held]
        geometry_lost = [name for name in geometry if name in lost and name != "mirrored"]
        assert grey_lost == ["gamma 0.5", "gamma 0.7", "gamma 2.0"] and degrade_held == ["noise 10", "speckle 0.25"]
        # the one interval of the published system that ends at zero to three decimals
        assert [name for name in CHANGED if at_zero(losses[name])] == ["gamma 2.0"]
        assert geometry_lost == ["zoomed out 0.8", "rotated 20 degrees"] and all(name in lost for name in time)
        grey_worst = min(losses[name]["value"] for name in grey)
        grey_mean = sum(-losses[name]["value"] for name in grey) / len(grey)
        grey_include = [name for name in grey if not below(losses[name])]
        degrade_lost = [name for name in degrade if name in lost]
        degrade_values = sorted(-losses[name]["value"] for name in degrade_lost)
        assert degrade_lost == ["noise 25", "blur 1 pixel", "blur 2 pixels", "JPEG quality 30", "JPEG quality 10", "112 pixels", "64 pixels"]
        # single recordings
        mirrored_wrong, clean_wrong = RECORDINGS - right(P, "mirrored"), RECORDINGS - right(P, CLEAN)
        classify = {name: P["conditions"][name]["classify"] for name in NAMES}
        by_errors = sorted(CHANGED, key=lambda name: -classify[name]["errors"])
        assert by_errors[:4] == ["mirrored", "noise 25", "speckle 0.25", "still frame"]
        assert classify["speckle 0.25"]["errors"] > classify["still frame"]["errors"] > max(classify[name]["errors"] for name in by_errors[4:])
        # the three other conditions of time: the multi streams lose, the pair streams and the single recordings hardly
        for name in ("shuffled frames", "played backwards", "half the frame rate"):
            assert below(losses[name]) and loss(P, name, "pairs")["interval"][1] >= 0 and right(P, name) >= right(P, CLEAN)
        by_classify = [classify[name]["errors"] for name in ("shuffled frames", "played backwards", "half the frame rate")]
        assert by_classify == [2, 1, 1] and classify[CLEAN]["errors"] == 1
        # the published system's intervals of other numbers that lie on the better side under a condition
        better = [(name, key) for name in CHANGED for _, part, key, _, _ in NUMBERS if key != "segment_f1_50" or part != "multi"
                  if (below(loss(P, name, part, key)) if key in LOWER_IS_BETTER else above(loss(P, name, part, key)))]
        assert len(better) == 9 and ("noise 10", "extra_segments") in better and ("speckle 0.25", "extra_segments") in better
        # a count of confused recordings has an interval exactly where both counts are above 5
        for name in NAMES:
            for key in ("a4c_as_a5c", "a5c_as_a4c"):
                counts = (value(P, name, "recordings", key), value(S, name, "recordings", key))
                assert (d["published_minus_stage2"][name]["recordings"][key]["interval"] is not None) == (min(counts) > 5)
                if name != CLEAN:
                    assert loss(P, name, "recordings", key)["interval"] is None and loss(S, name, "recordings", key)["interval"] is None

        # stage 2, and the published system against it
        stage2_setting = S["setting"]
        assert stage2_setting["status"] == "feasible" and stage2_setting["floors_dropped"] is False
        every = d["published_minus_stage2"]
        apart = {name: every[name]["multi"]["segment_f1_50"] for name in NAMES}
        ahead = [name for name in CHANGED if above(apart[name])]
        assert [name for name in NAMES if below(apart[name])] == [] and not above(apart[CLEAN])
        ahead_sorted = sorted(ahead, key=lambda name: -apart[name]["value"])
        assert ahead_sorted[:5] == ["noise 25", "blur 2 pixels", "64 pixels", "speckle 0.25", "contrast 0.5"]
        assert apart[ahead_sorted[4]]["value"] > 1.9 * apart[ahead_sorted[5]]["value"]  # the five stand apart from the rest
        rest = ahead_sorted[5:]
        degrade_ahead, grey_ahead = [name for name in degrade if name in ahead], [name for name in grey if name in ahead]
        assert degrade_ahead == ["noise 10", "noise 25", "speckle 0.25", "blur 2 pixels", "64 pixels"] and grey_ahead == ["darker by 40", "contrast 0.5"]
        assert [name for name in NAMES if at_zero(apart[name])] == ["half the frame rate"]
        assert sorted(rest) == sorted(["noise 10", "half the frame rate", "darker by 40"])
        stage2_losses = {name: loss(S, name) for name in CHANGED}
        assert below(stage2_losses["mirrored"]) and not above(apart["mirrored"]) and not below(apart["mirrored"])
        # the intervals of the other numbers that lie on the stage-2 ensemble's side
        for_stage2 = [(name, key) for name in NAMES for _, part, key, _, _ in NUMBERS
                      if (above(every[name][part][key]) if key in LOWER_IS_BETTER else below(every[name][part][key]))]
        under = sorted({name for name, _ in for_stage2})
        assert len(for_stage2) == 9 and under == ["gamma 1.5", "gamma 2.0", "mirrored", "still frame"]
        for name in ("mirrored", "still frame"):
            assert {key for n, key in for_stage2 if n == name} >= {"cut_precision", "extra_segments", "cuts_per_recording"}

        # the ninth view
        val, train = P["ninth_view"]["validation"], P["ninth_view"]["training"]
        assert val["given"]["PSAX"] == val["recordings"] == 46 and train["recordings"] == 319
        assert {view: train["given"][view] for view in VIEWS if view != "PSAX" and train["given"][view]} == {"A4C": 1}
        reader = P["reader_against_benchmark_frames"]
        assert reader["recordings"] == 521 and reader["recordings_given_another_view"] == 0 and reader["largest_difference_in_a_probability"] < 1e-6
        assert S["reader_against_benchmark_frames"]["recordings_given_another_view"] == 0
        sure = {label: P["ninth_view"][label]["known_above_ninth"] for label in ("validation", "training")}
        includes_half = {label: [key for key in ("probability", "margin", "negative_entropy")
                                 if sure[label][key]["interval"][0] <= 0.5 <= sure[label][key]["interval"][1]] for label in sure}
        assert includes_half == {"validation": ["margin"], "training": ["probability", "margin", "negative_entropy"]}
        assert sure["validation"]["probability"]["interval"][0] > 0.5 and sure["validation"]["negative_entropy"]["interval"][0] > 0.5
        stage2_sure = {label: S["ninth_view"][label]["known_above_ninth"]["probability"] for label in ("validation", "training")}
        assert all(stage2_sure[label]["interval"][0] > 0.5 for label in sure)
        known = val["known"]
        assert known == train["known"] and known["recordings"] == 521

        def interval(entry):
            return f"{f3(entry['value'])} [{f3(entry['interval'][0])}, {f3(entry['interval'][1])}]"

        # the cells of the three tables of every number that a tie decides: as they stood before the correction of 2026-11-28, and as they stand
        def before(x, digits):
            return f"{Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP):+.{digits}f}"

        tables = [(PUBLISHED, {name: P["conditions"][name]["minus_unchanged"] for name in CHANGED}),
                  (STAGE2, {name: S["conditions"][name]["minus_unchanged"] for name in CHANGED}), ("published minus stage 2", d["published_minus_stage2"])]
        moved = [(title, name, key, ("the value", "the lower end", "the upper end")[place], before(scale * x, digits),
                  rounded(scale * x, digits, True, counted=False))
                 for title, entries in tables for name in entries for _, part, key, digits, scale in NUMBERS
                 for place, x in enumerate((entries[name][part][key]["value"], *(entries[name][part][key]["interval"] or ())))
                 if before(scale * x, digits) != rounded(scale * x, digits, True, counted=False)]

        lines = [
            "### Part 2: forward passes",
            "",
            "**Run on 2026-11-26 from 03:36 to 07:43 UTC and from 20:47 to 22:19 UTC with the code of commit `fb674f6`, which is the plan.** The "
            "server stopped in "
            "between, and a condition whose file existed was not computed again. The output of the report is "
            "[`ablations/forward.json`](ablations/forward.json). The "
            "unchanged frames, run again, gave the stored outputs to less than 0.000001 in every probability and the published number of the "
            "selection half, 0.9594.",
            "",
            "The streams and the single recordings are those of the selection half. The recordings of the ninth view are two lists of their own, and "
            "they are set "
            "against the 521 single recordings of both halves. Intervals are at 95% and draw streams for the numbers about streams and recordings "
            "for the numbers "
            "about recordings. They are many, and none is corrected for their number. An end printed as +0.000 or -0.000 can be a number just off "
            "zero; the file has "
            "every digit. Part 2 describes and gets no word.",
            "",
            "**The published system under every condition.**",
            "",
            *system_rows(P),
            "",
            f"- **A mirrored image is the largest loss on the multi streams.** Segment F1 falls from {f3(value(P, CLEAN))} to "
            f"{f3(value(P, 'mirrored'))}, which is "
            f"{signed(mirrored)}; the next largest loss is {f3(-still['value'])}. By the majority rule {mirrored_wrong} of the {RECORDINGS} single "
            "recordings are read "
            f"wrong, where {clean_wrong} are wrong on the unchanged frames, and by `classify`'s rule {classify['mirrored']['errors']}, where "
            f"{classify[CLEAN]['errors']} is. Their macro-F1 falls from {f3(value(P, CLEAN, 'recordings', 'macro_f1'))} to "
            f"{f3(value(P, 'mirrored', 'recordings', 'macro_f1'))}. No training stage mirrored a frame; part 2 does not test whether that is the "
            "cause. The stage-2 "
            f"ensemble loses {f3(-stage2_losses['mirrored']['value'])} under it ({signed(stage2_losses['mirrored'])}).",
            f"- **Windows that show one frame are the second largest.** When every window shows its centre frame sixteen times, the multi streams lose "
            f"{f3(-still['value'])} ({signed(still)}). Of the single recordings {right(P, 'still frame')} of {RECORDINGS} are right by the majority "
            "rule, where "
            f"{right(P, CLEAN)} are on the unchanged frames, and {classify['still frame']['errors']} are wrong by `classify`'s rule, where "
            f"{classify[CLEAN]['errors']} is. Inside the single recordings the decoded view changes {changes(P, 'still frame')} times, where it changes "
            f"{changes(P, CLEAN)} times, the mean probability falls from {f3(classify[CLEAN]['mean_probability'])} to "
            f"{f3(classify['still frame']['mean_probability'])}, and the seeds differ in {classify['still frame']['seeds_differ']} recordings, where "
            "they differ in "
            f"{classify[CLEAN]['seeds_differ']}.",
            f"- **Shuffled frames, frames played backwards and half the frame rate each cost the multi streams something.** They cost "
            f"{f3(-losses['shuffled frames']['value'])}, {f3(-losses['played backwards']['value'])} and "
            f"{f3(-losses['half the frame rate']['value'])}, and each "
            "interval lies below zero. Only for the shuffled frames is the order the one thing that changes, with the marker staying on the centre frame "
            "(section 4). A stream played backwards also has its windows at other places, and half the frame rate puts a cut up to 0.1 s late by "
            "construction. For "
            "the pair streams no interval of the three lies below zero. By the majority rule no more single recordings are read wrong than on the "
            "unchanged frames. "
            f"By `classify`'s rule {by_classify[0]} are wrong under the shuffled frames, where {classify[CLEAN]['errors']} is wrong on the unchanged "
            "frames and under "
            "the other two.",
            f"- **Grey levels.** The interval lies below zero for gamma 0.5 and gamma 0.7, and the larger of the two losses is {f3(-grey_worst)}. "
            "For gamma 2.0 "
            f"it ends at zero to three decimals ({signed(losses['gamma 2.0'])}). For the other five conditions of grey levels it includes zero.",
            f"- **Degradation.** Noise with a spread of 25 grey levels, blur of 1 and of 2 pixels, JPEG at quality 30 and 10, and 112 and 64 pixels cost "
            f"{f3(degrade_values[0])} to {f3(degrade_values[-1])}, each with an interval below zero. For noise with a spread of 10 and for speckle "
            "the interval "
            f"includes zero. For the single recordings the strong noise and the speckle are harder than the multi streams show: `classify` reads "
            f"{classify['noise 25']['errors']} and {classify['speckle 0.25']['errors']} wrong under them.",
            f"- **Geometry other than mirroring.** The interval lies below zero for {listed(geometry_lost)} ({f3(-losses['zoomed out 0.8']['value'])} and "
            f"{f3(-losses['rotated 20 degrees']['value'])}), and includes zero for zoomed in 1.2 and rotated 10 degrees.",
            f"- **For segment F1 on the multi streams no interval lies above zero.** Of the 26 conditions, {len(lost) - 1} have an interval below "
            "zero, one has an "
            f"interval that ends at zero to three decimals, and {len(held)} have one that includes zero. Under {len(higher)} conditions the value is "
            "above that of "
            f"the unchanged frames, by {f3(higher[0])} to {f3(higher[-1])}. After the mirrored image and the still frame the largest loss is "
            f"{f3(-third)}. Other "
            f"numbers do have intervals on the better side under a condition, {len(better)} in all: for example fewer segments that match nothing "
            "under noise with a "
            "spread of 10 and under speckle (the tables of every number).",
            "",
            "**The stage-2 ensemble under every condition.** Its decoder setting is the one that the rule gives it on the calibration half: half window "
            f"{stage2_setting['params']['half_window']}, threshold {f3(stage2_setting['params']['threshold'])}, the "
            f"{stage2_setting['params']['threshold_quantile']:g} "
            "quantile. The floors were kept.",
            "",
            *system_rows(S),
            "",
            "**The published system minus the stage-2 ensemble,** with the same draws.",
            "",
            *difference_rows(),
            "",
            f"- **On the unchanged frames the stage-2 ensemble has the higher value.** It reaches {f3(value(S, CLEAN))} where the published system reaches "
            f"{f3(value(P, CLEAN))}: {signed(apart[CLEAN])}, an interval that includes zero.",
            f"- **Under five conditions the published system's segment F1 is the higher by {f3(apart['contrast 0.5']['value'])} to "
            f"{f3(apart['noise 25']['value'])}.** Noise with a spread of 25 grey levels {signed(apart['noise 25'])}, "
            f"blur of 2 pixels {signed(apart['blur 2 pixels'])}, 64 pixels {signed(apart['64 pixels'])}, speckle {signed(apart['speckle 0.25'])} and "
            "a contrast of "
            f"0.5 {signed(apart['contrast 0.5'])}. There the stage-2 ensemble falls to {f3(value(S, 'noise 25'))}, {f3(value(S, 'blur 2 pixels'))}, "
            f"{f3(value(S, '64 pixels'))}, {f3(value(S, 'speckle 0.25'))} and {f3(value(S, 'contrast 0.5'))}.",
            f"- **Two more intervals lie above zero:** noise 10 ({signed(apart['noise 10'])}) and darker by 40 ({signed(apart['darker by 40'])}). For half the "
            f"frame rate the interval begins at zero to three decimals ({signed(apart['half the frame rate'])}).",
            f"- **For segment F1 on the multi streams no interval lies below zero.** Under the other {len(CHANGED) - len(ahead)} conditions it "
            "includes zero, the "
            f"mirrored image among them. Other numbers do have intervals on the side of the stage-2 ensemble, {len(for_stage2)} in all. Under the "
            "mirrored image "
            "and under the still frame the published system has the lower cut precision, more segments that match nothing and more label changes inside single "
            "recordings (the last table of part 2).",
            "- **What this does not say.** Stage 3 is 1,500 further updates and its input changes together. Part 2 cannot tell the two apart "
            "(section 4). Part 4 is "
            "planned for that (section 12).",
            "",
            "**A view the system does not know.** The recordings of EV9V's ninth view, read as `echoview classify --fps 30` reads a folder.",
            "",
            *ninth_rows(),
            "",
            "The chance that a recording of the five views is surer than a recording of the ninth view, for three numbers. 0.5 means that the number "
            "does not tell "
            f"them apart. The {known['recordings']} single recordings of the five views were read by the same reader. The intervals draw recordings.",
            "",
            *apart_rows(),
            "",
            f"- **The published system names the ninth view PSAX.** All {val['recordings']} recordings of the validation split and "
            f"{train['given']['PSAX']} of the "
            f"{train['recordings']} of the training split get that name, and one is named A4C. {val['probability']['above_0.9']} of the "
            f"{val['recordings']} and "
            f"{train['probability']['above_0.9']} of the {train['recordings']} have a probability above 0.9, with a median of "
            f"{f3(val['probability']['median'])} and "
            f"{f3(train['probability']['median'])}. The recordings of the five views have a median of {f3(known['median_probability'])}.",
            f"- **The probability tells little.** For the validation list the chance is {interval(sure['validation']['probability'])} and for the "
            "larger training "
            f"list {interval(sure['training']['probability'])}. The margin and the negative entropy give nearly the same numbers. On the training "
            "list every interval "
            "includes 0.5. On the validation list that of the margin does, and the other two lie just above it. Section 2 quoted the 0.575 from the "
            "research records "
            "with an interval of 0.485 to 0.642, which includes 0.5. The two intervals are not the same, and this document does not hold how the "
            "records drew theirs.",
            f"- **For the stage-2 ensemble the chance is** {interval(stage2_sure['validation'])} and {interval(stage2_sure['training'])}. Both "
            "intervals lie above "
            "0.5.",
            f"- **One reader for both kinds.** The {reader['recordings']} single recordings of the five views, read as a folder, get the view that "
            "the benchmark's own "
            "frames give them in every case, and no probability differs by more than 0.000001.",
            "",
            "**Every number of section 5.** The tables above show a part. Here is every number: for the published system and for the stage-2 ensemble each "
            "condition minus the unchanged frames, and then the published system minus the stage-2 ensemble. A count of confused recordings has an "
            "interval only "
            "where both counts are above 5.",
            "",
            "The published system, each condition minus the unchanged frames:",
            "",
            *every_number_rows({name: P["conditions"][name]["minus_unchanged"] for name in CHANGED}, CHANGED),
            "",
            "The stage-2 ensemble, each condition minus the unchanged frames:",
            "",
            *every_number_rows({name: S["conditions"][name]["minus_unchanged"] for name in CHANGED}, CHANGED),
            "",
            "The published system minus the stage-2 ensemble:",
            "",
            *every_number_rows(every, NAMES),
            "",
            "**Against what was expected** (section 7).",
            "",
            f"- **Grey levels.** Small losses were expected. The largest is {f3(-grey_worst)}, the mean over the eight conditions is {f3(grey_mean)}, and for "
            f"{len(grey_include)} of the eight the interval includes zero.",
            "- **Strong noise, 64 pixels and JPEG at quality 10.** Larger losses than under grey levels were expected. They cost "
            f"{f3(-losses['noise 25']['value'])}, "
            f"{f3(-losses['64 pixels']['value'])} and {f3(-losses['JPEG quality 10']['value'])}, each with an interval below zero. A gamma of 0.5 alone costs "
            f"{f3(-losses['gamma 0.5']['value'])}, more than the JPEG.",
            "- **The mirrored image** was named as the open one. It is the largest loss on the multi streams.",
            "- **The still frame.** Single recordings read nearly as well and streams read worse were expected. The streams lose "
            f"{f3(-still['value'])}. By the "
            f"majority rule {right(P, CLEAN) - right(P, 'still frame')} more recordings are wrong, and by `classify`'s rule "
            f"{classify['still frame']['errors'] - classify[CLEAN]['errors']} more.",
            f"- **Shuffled frames** were expected to cost little. They cost {f3(-losses['shuffled frames']['value'])}, about what a gamma of 0.5 "
            "costs, with an "
            "interval below zero.",
            f"- **Stage 3 against stage 2.** The published system was expected to have the higher value under changes of grey levels and of image quality. Its "
            f"interval lies above zero under {len(degrade_ahead)} of the 9 conditions of degradation and under {len(grey_ahead)} of the 8 of grey levels. No "
            "difference on the unchanged frames was expected, and none is shown.",
            "- **The ninth view.** Nearly all recordings named PSAX and most with a probability above 0.9 were expected, and that is so. A "
            "separation near 0.6 was "
            f"expected. It is {f3(sure['validation']['probability']['value'])} on the validation list and "
            f"{f3(sure['training']['probability']['value'])} on the "
            "training list.",
            "",
            "**What follows.** By section 6 a weakness under a condition is reported as a limit. The model card and the README now name two: "
            "mirrored images, and "
            "windows that show one frame. Training against a mirrored image would make another model, with a plan of its own. Section 12 is that plan.",
            "",
            "**Corrected on 2026-11-28.** Two cells of the tables of every number were ties of the last digit shown, and they had been rounded down because "
            "the file stores them a hair below the tie. For the stage-2 ensemble zoomed in to 1.2, the Edit score minus the unchanged frames is "
            f"{moved[0][5]} where {moved[0][4]} stood. For the published system minus the stage-2 ensemble rotated by 10 degrees, the upper end of the "
            f"interval of the cut recall is {moved[1][5]} where {moved[1][4]} stood. A tie is rounded away from zero in every cell, as the other cells "
            "were. No other number of parts 2 and 3 changes.",
            "",
        ]
        assert moved == [("stage 2", "zoomed in 1.2", "edit", "the value", "+0.2", "+0.3"),
                         ("published minus stage 2", "rotated 10 degrees", "cut_recall", "the upper end", "+0.012", "+0.013")]
        # "No other number of parts 2 and 3 changes": of all the numbers that part 2 prints, the rule moves these two cells and nothing else
        assert sorted(ties) == sorted([(S["conditions"]["zoomed in 1.2"]["minus_unchanged"]["multi"]["edit"]["value"], 1),
                                       (d["published_minus_stage2"]["rotated 10 degrees"]["multi"]["cut_recall"]["interval"][1], 3)])
        assert len(rest) == 3 and len(grey) - len(grey_lost) == 5 and (len(degrade), len(grey)) == (9, 8) and len(grey_include) == 5
        assert -losses["noise 25"]["value"] > -losses["gamma 0.5"]["value"] > -losses["JPEG quality 10"]["value"]
        assert abs(-losses["gamma 0.5"]["value"] - (-grey_worst)) < 1e-12 and abs(-losses["shuffled frames"]["value"] - (-grey_worst)) < 0.003
        assert -mirrored["value"] > 1.5 * -still["value"]
        return "\n".join(lines)

    return text()


def part_three(d: dict, part1: dict) -> str:
    """Section 11: what part 1 left open. `part1` is the output of part 1, to which part 3 is held."""
    KEPT_SETTING, BY_THE_RULE = "the temperature, the published setting", "the temperature, the setting by the rule"
    RECORDINGS = 284


    ties = []  # the numbers that are a tie of the last digit shown only because they are taken at twelve decimals first


    def rounded(x, digits, sign=False, counted=True):
        """A number at the digits shown, with a tie rounded away from zero as in ordinary rounding (Python's own formatting
        rounds a tie of the binary number to even). The number is taken at twelve decimals first, so that a tie which the
        file stores a hair off itself is the tie."""
        step = Decimal(1).scaleb(-digits)
        number = Decimal(repr(float(x)))
        value = number.quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN).quantize(step, rounding=ROUND_HALF_UP)
        if counted and value != number.quantize(step, rounding=ROUND_HALF_UP):
            ties.append((float(x), digits))
        return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


    def f3(x):
        return rounded(x, 3)


    def signed(entry, digits=3, scale=1.0):
        if entry.get("interval") is None:
            return rounded(scale * entry["value"], digits, True)
        low, high = entry["interval"]
        ends = f"[{rounded(scale * low, digits, True)}, {rounded(scale * high, digits, True)}]"
        return f"{rounded(scale * entry['value'], digits, True)} {ends}"


    def three(r):
        return f"{f3(r['segment_f1_50_multi'])} | {f3(r['native_macro_f1'])} | {f3(r['boundary_recall_multi'])}"


    def whole(x):
        assert abs(x - round(x)) < 1e-6, x
        return round(x)


    def listed(names):
        names = list(names)
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


    # ---------------------------------------------------------------------------------- thresholds above the grid
    higher = d["higher_thresholds"]
    table = higher["settings"]


    def added_rows():
        rows = ["| Half window | Quantile | Threshold | Calibration half: segment F1 | Macro-F1 | Boundary recall | Selection half: segment F1 | "
            "Macro-F1 | Boundary recall |",
                "|---|---|---|---|---|---|---|---|---|"]
        for s in table:
            if s["added"] or s["threshold_quantile"] == 0.999:
                rows.append(f"| {s['half_window']} | {s['threshold_quantile']:g} | {rounded(s['threshold'], 3)} | {three(s['calibration'])} | "
                    f"{three(s['selection'])} |")
        return rows


    def higher_text():
        rule = higher["rule"]
        kept = [{k: s[k] for k in ("half_window", "threshold", "threshold_quantile", "calibration", "selection")} for s in table if not s["added"]]
        before = [{k: s[k] for k in ("half_window", "threshold", "threshold_quantile", "calibration", "selection")}
                  for s in part1["decoder_settings"]["settings"]]
        assert kept == before and len(table) == 55  # the 40 settings of part 1 come out again, to the last digit
        chosen = table[rule["chosen"]]
        assert (chosen["half_window"], chosen["threshold_quantile"], chosen["added"]) == (7, 0.999, False) and rule["is_the_published_setting"] is True
        assert higher["word"] == "the published setting stays" and rule["floors_dropped"] is False and rule["is_at_the_highest_quantile"] is False
        published = chosen["selection"]["segment_f1_50_multi"]
        added = [s for s in table if s["added"]]
        assert higher["added_on_selection"] == {"settings": 15, "above_the_published": 0, "highest": max(s["selection"]["segment_f1_50_multi"] for s in added)}
        best_added = max(added, key=lambda s: s["selection"]["segment_f1_50_multi"])
        assert (best_added["half_window"], best_added["threshold_quantile"]) == (7, 0.9995)
        on_calibration = max(table, key=lambda s: (s["calibration"]["segment_f1_50_multi"], -s["order"]))
        assert on_calibration is chosen  # the published setting has the highest segment F1 of all 55 on the calibration half as well
        assert all(s["calibration"]["segment_f1_50_multi"] < chosen["calibration"]["segment_f1_50_multi"] for s in added)
        falls = all(a["selection"]["segment_f1_50_multi"] > b["selection"]["segment_f1_50_multi"]
                    for half_window in (5, 7, 9, 11, 15)
                    for a, b in zip([s for s in table if s["half_window"] == half_window and s["threshold_quantile"] >= 0.999][:-1],
                                    [s for s in table if s["half_window"] == half_window and s["threshold_quantile"] >= 0.999][1:]))
        assert falls  # on the selection half every step above the 0.999 quantile lowers the value, for every half window
        last = [s for s in added if s["threshold_quantile"] == 1.0]
        at = {(s["half_window"], s["threshold_quantile"]): s["calibration"]["segment_f1_50_multi"] for s in table}
        rises = [half_window for half_window in (5, 7, 9, 11, 15) if at[(half_window, 0.9995)] > at[(half_window, 0.999)]]
        assert rises == [5, 9, 11]
        lowest_recall = min(s["selection"]["boundary_recall_multi"] for s in last)
        infeasible = rule["n_settings"] - rule["n_feasible"]
        assert all(s["added"] for s in table if not (s["calibration"]["native_macro_f1"] >= 0.980 and s["calibration"]["boundary_recall_multi"] >= 0.95))
        apart = higher["selection"]["chosen_minus_published"]["multi"]["segment_f1_50"]
        assert apart["value"] == 0.0
        return [
            f"- **The published setting stays.** Over the 55 settings the rule takes half window 7 at the 0.999 quantile again. {rule['n_feasible']} "
            "of the 55 keep "
            f"the floors on the calibration half, and the {infeasible} that do not are all added ones.",
            "- **No added setting is above the published one, on either half.** On the selection half the highest of the 15 is "
            f"{f3(best_added['selection']['segment_f1_50_multi'])}, "
            f"at half window 7 and the 0.9995 quantile, where the published setting has {f3(published)}. For every half window each step above the "
            "0.999 quantile "
            "lowers the value on the selection half.",
            f"- **At quantile 1 the decoder misses cuts.** There the boundary recall on the selection half is {f3(lowest_recall)} to "
            f"{f3(max(s['selection']['boundary_recall_multi'] for s in last))}.",
            "- **What this answers.** Part 1 could not say what a higher threshold would do. No added setting is above the published one on either "
            "half. On the "
            f"calibration half the first step up is a little higher than the 0.999 quantile for half windows {listed(str(h) for h in rises)}, and "
            "still below the "
            "published setting. This was read on the two halves on which the decoder was developed.",
        ]


    # ---------------------------------------------------------------------------------------- the halves exchanged
    exchanged = d["halves_exchanged"]


    def exchanged_text():
        rule, rows = exchanged["rule"], exchanged["settings"]
        chosen, highest = rows[rule["chosen"]], rows[exchanged["highest_on_selection_whatever_the_floors"]]
        before = part1["decoder_settings"]["halves_exchanged"]
        assert {k: chosen[k] for k in before["setting"]} == before["setting"] and chosen["calibration"] == before["calibration"]["chosen_on_selection"]
        assert rule["floors_dropped"] is False and rule["n_settings"] == 40 and highest["feasible_on_selection"] is False
        assert highest["selection"]["native_macro_f1"] < 0.980 and highest["selection"]["boundary_recall_multi"] >= 0.95  # the macro-F1 floor alone removes it
        higher_than_chosen = [r for r in rows if r["selection"]["segment_f1_50_multi"] > chosen["selection"]["segment_f1_50_multi"]]
        assert higher_than_chosen and not any(r["feasible_on_selection"] for r in higher_than_chosen)
        by_macro = [r for r in higher_than_chosen if r["selection"]["native_macro_f1"] < 0.980]
        by_recall = [r for r in higher_than_chosen if r["selection"]["boundary_recall_multi"] < 0.95]
        assert len(by_macro) + len(by_recall) == len(higher_than_chosen)  # each of them misses one floor and keeps the other
        assert all(r["threshold"] > chosen["threshold"] for r in higher_than_chosen)
        return [
            f"- **{rule['n_feasible']} of the 40 settings keep the floors on the selection half.** The rule takes half window "
            f"{chosen['half_window']} with a threshold "
            f"of {f3(chosen['threshold'])}, as part 1 reported. On the selection half that setting has "
            f"{f3(chosen['selection']['segment_f1_50_multi'])}, and on the "
            f"calibration half {f3(chosen['calibration']['segment_f1_50_multi'])}.",
            f"- **The setting with the highest value on the selection half does not keep the floors.** It is half window {highest['half_window']} "
            "with a threshold of "
            f"{f3(highest['threshold'])}: segment F1 {f3(highest['selection']['segment_f1_50_multi'])}, and a macro-F1 of single recordings of "
            f"{f3(highest['selection']['native_macro_f1'])}, below the floor of 0.980.",
            f"- **What this answers.** {len(higher_than_chosen)} settings have a higher value on the selection half than the rule's choice, and each "
            "has a higher "
            f"threshold. None keeps the floors there: {len(by_macro)} fall below the floor for the macro-F1 of single recordings and "
            f"{len(by_recall)} below the "
            "floor for the boundary recall. So with the halves exchanged it is the two floors that send the rule to a lower threshold. Part 1 could "
            "only suggest "
            "that.",
        ]


    def exchanged_rows():
        rows = ["| Half window | Quantile | Threshold | Selection half: segment F1 | Macro-F1 | Boundary recall | Keeps the floors | "
                "Calibration half: segment F1 | Macro-F1 | Boundary recall |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in exchanged["settings"]:
            rows.append(f"| {s['half_window']} | {s['threshold_quantile']:g} | {rounded(s['threshold'], 5)} | {three(s['selection'])} | "
                        f"{'yes' if s['feasible_on_selection'] else 'no'} | {three(s['calibration'])} |")
        return rows


    # ------------------------------------------------------------------------------------------------ the view test
    view = d["view_test"]
    KIND = {"native": "single recordings", "pairs": "pair streams", "multi": "multi streams"}


    def view_rows():
        rows = ["| Half | Kind | Streams | Peaks | Dropped by the view test | Streams with a dropped peak | Streams in which a sample changes | "
            "Samples that change |",
                "|---|---|---|---|---|---|---|---|"]
        for half in ("calibration", "selection"):
            for kind, name in KIND.items():
                c = view[half][kind]
                rows.append(f"| {half} | {name} | {c['streams']} | {c['peaks']} | {c['dropped']} | {c['streams_with_a_dropped_peak']} | "
                    f"{c['streams_that_change']} | "
                            f"{c['samples_that_change']} |")
        return rows


    def view_text():
        total = {half: {key: sum(view[half][kind][key] for kind in KIND) for key in ("peaks", "dropped", "samples_that_change", "streams_that_change")}
                 for half in view}
        assert total["selection"]["samples_that_change"] == 0 and view["selection"]["native"]["dropped"] == view["calibration"]["native"]["dropped"] == 0
        changed = [(half, kind) for half in view for kind in KIND if view[half][kind]["samples_that_change"]]
        assert changed == [("calibration", "pairs")]
        one = view["calibration"]["pairs"]
        assert 25 < (total["selection"]["peaks"] + total["calibration"]["peaks"]) / (total["selection"]["dropped"] + total["calibration"]["dropped"]) < 35
        return [
            f"- **The test drops peaks, and almost never a sample changes.** On the selection half it drops {total['selection']['dropped']} of "
            f"{total['selection']['peaks']} peaks and on the calibration half {total['calibration']['dropped']} of {total['calibration']['peaks']}. "
            "None is inside a "
            "single recording.",
            "- **On the selection half no sample changes,** which is why part 1 saw every number unchanged. On the calibration half "
            f"{one['streams_that_change']} "
            f"pair stream changes, in {one['samples_that_change']} samples.",
            "- **What this answers.** The test is not idle: it removes about one peak in thirty. A removed peak has the same view on both sides, and "
            "the pieces "
            "beside it then take the same view anyway, so the views of the samples hardly ever depend on the test.",
        ], total


    # ---------------------------------------------------------------------------------- the other decoders' numbers
    others = d["other_decoders"]


    def other_rows():
        rows = ["| Decoder | Its number | Calibration half: segment F1 | Macro-F1 | Boundary recall | Selection half: segment F1 | Macro-F1 | "
            "Boundary recall | Chosen |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name, o in others.items():
            for r in o["values"]:
                rows.append(f"| {name} | {r['value']:g} | {three(r['calibration'])} | {three(r['selection'])} | "
                    f"{'yes' if r['order'] == o['rule']['chosen'] else ''} |")
        return rows


    def summary_rows():
        rows = ["| Decoder | Values | Chosen | Segment F1 of the chosen | Lowest | Highest | Within 0.01 of the chosen | Within 0.02 | Above the chosen |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name, o in others.items():
            s = o["selection_segment_f1_50_multi"]
            rows.append(f"| {name} | {s['values']} | {o['chosen']:g} | {f3(s['of_the_chosen'])} | {f3(s['lowest'])} | {f3(s['highest'])} | "
                f"{s['within_0.01_of_the_chosen']} | "
                        f"{s['within_0.02_of_the_chosen']} | {s['above_the_chosen']} |")
        return rows


    def other_text():
        for name, o in others.items():  # the rule chose what part 1 reported for each
            assert o["chosen"] == part1["decoders"]["others"][name]["chosen"]["value"] and o["rule"]["floors_dropped"] is False
            assert o["selection_segment_f1_50_multi"]["of_the_chosen"] == part1["decoders"]["others"][name]["selection"]["multi"]["segment_f1_50"]["value"]
        viterbi, mean, mode, duration = others["Viterbi path"], others["mean filter"], others["mode filter"], others["minimum duration"]
        v = viterbi["selection_segment_f1_50_multi"]
        near = [r["value"] for r in viterbi["values"] if abs(r["selection"]["segment_f1_50_multi"] - v["of_the_chosen"]) <= 0.02]
        assert near == [8.0, 16.0] and v["within_0.02_of_the_chosen"] == 2
        by_cost = {r["value"]: r["selection"]["segment_f1_50_multi"] for r in viterbi["values"]}
        m = mean["selection_segment_f1_50_multi"]
        mean_best = max(mean["values"], key=lambda r: r["selection"]["segment_f1_50_multi"])
        mode_best = max(mode["values"], key=lambda r: r["selection"]["segment_f1_50_multi"])
        assert mean_best["value"] == mode_best["value"] == 21 and duration["selection_segment_f1_50_multi"]["above_the_chosen"] == 0
        return [
            f"- **Two of the Viterbi path's eight costs lie within 0.02 of the chosen one:** 8 and 16, with {f3(by_cost[8.0])} and "
            f"{f3(by_cost[16.0])}. A cost of "
            f"4 gives {f3(by_cost[4.0])} and a cost of 32 gives {f3(by_cost[32.0])}. Its grid doubles the cost at every step, which is coarse.",
            f"- **For the two filters a larger window does better on the selection half than the chosen one.** At 21 samples the mean filter has "
            f"{f3(mean_best['selection']['segment_f1_50_multi'])} and the mode filter {f3(mode_best['selection']['segment_f1_50_multi'])}, where the "
            "chosen windows "
            f"have {f3(m['of_the_chosen'])} and {f3(mode['selection_segment_f1_50_multi']['of_the_chosen'])}. These values were not chosen on the "
            "calibration half, "
            "and by the plan they are not set against the released decoder.",
            "- **The minimum duration has no value above the chosen one,** and "
            f"{duration['selection_segment_f1_50_multi']['within_0.02_of_the_chosen']} of its 7 "
            "values lie within 0.02 of it.",
        ]


    # ------------------------------------------------------------------------------------------- lower temperatures
    temperature = d["temperature"]


    def curve_rows():
        curve = {c["temperature"]: c["log_loss"] for c in temperature["calibration_curve"]}
        shown = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00)
        return ["| Temperature | " + " | ".join(f"{rounded(t, 2)}" for t in shown) + " |", "|---|" + "---|" * len(shown),
                "| Log loss, calibration half | " + " | ".join(f"{rounded(curve[t], 4)}" for t in shown) + " |"]


    def temperature_rows():
        rows = ["| The 284 single recordings of the selection half | Wrong | Mean probability | Right | Mean probability minus the share that is "
            "right | Calibration error, "
                "equal width | Calibration error, equal count | Brier score | Log loss |", "|---|---|---|---|---|---|---|---|---|"]
        for name, e in temperature["selection"].items():
            gap = e["probability_minus_right"]
            label = name if name == "as published" else f"with a temperature of {rounded(e['temperature'], 2)}"
            rows.append(f"| {label} | {e['errors']} | {f3(e['mean_probability'])} | {round(RECORDINGS * e['share_right'])} | {rounded(gap['value'], 3, True)} "
                f"[{rounded(gap['interval'][0], 3, True)}, "
                        f"{rounded(gap['interval'][1], 3, True)}] | {f3(e['ece'])} | {f3(e['ece_equal_count'])} | {f3(e['brier']['value'])} | "
                        f"{f3(e['log_loss']['value'])} |")
        return rows


    def stream_rows():
        s = temperature["streams"]
        rows = ["| System | Decoder setting | Segment F1 at 50%, multi streams | Minus as published | Pair streams | Wrong samples | Cut precision | "
            "Cut recall | "
                "Segments that match nothing | Recordings right of 284, majority rule |", "|---|---|---|---|---|---|---|---|---|---|"]
        setting = s["setting_by_the_rule"]

        def cells(numbers):
            m = numbers["multi"]
            right = whole(RECORDINGS * numbers["recordings"]["accuracy"]["value"])
            return (f"{f3(numbers['pairs']['segment_f1_50']['value'])} | {rounded(100 * m['wrong_samples']['value'], 1)}% | "
                    f"{f3(m['cut_precision']['value'])} | {f3(m['cut_recall']['value'])} | {rounded(m['extra_segments']['value'], 0)} | {right}")

        rows.append(f"| as published | half window 7, threshold 0.174 | {f3(s['published']['multi']['segment_f1_50']['value'])} | | {cells(s['published'])} |")
        rows.append(f"| {KEPT_SETTING} | half window 7, threshold 0.174 | {f3(s[KEPT_SETTING]['selection']['multi']['segment_f1_50']['value'])} | "
                    f"{signed(s[KEPT_SETTING]['minus_published']['multi']['segment_f1_50'])} | {cells(s[KEPT_SETTING]['selection'])} |")
        rows.append(f"| {BY_THE_RULE} | half window {setting['half_window']}, threshold {f3(setting['threshold'])} | "
                    f"{f3(s[BY_THE_RULE]['selection']['multi']['segment_f1_50']['value'])} | "
                    f"{signed(s[BY_THE_RULE]['minus_published']['multi']['segment_f1_50'])} | "
                    f"{cells(s[BY_THE_RULE]['selection'])} |")
        return rows


    NUMBERS = (  # the row's name, the part, the key, the digits, the scale: every number of section 5
        ("Segment F1 at 50%, multi streams", "multi", "segment_f1_50", 3, 1.0), ("Edit", "multi", "edit", 1, 1.0),
        ("Wrong samples, percentage points", "multi", "wrong_samples", 1, 100.0), ("Cut precision", "multi", "cut_precision", 3, 1.0),
        ("Cut recall", "multi", "cut_recall", 3, 1.0), ("Mean distance of matched cuts, seconds", "multi", "cut_distance_mean_s", 3, 1.0),
        ("Cuts at joins of the same view, percentage points", "multi", "same_view_cut_rate", 1, 100.0),
        ("Segments that match nothing", "multi", "extra_segments", 0, 1.0), ("Segments missed", "multi", "missed_segments", 0, 1.0),
        ("Segment F1 at 50%, pair streams", "pairs", "segment_f1_50", 3, 1.0), ("Accuracy, single recordings", "recordings", "accuracy", 3, 1.0),
        ("Macro-F1, single recordings", "recordings", "macro_f1", 3, 1.0), ("A4C read as A5C, recordings", "recordings", "a4c_as_a5c", 0, 1.0),
        ("A5C read as A4C, recordings", "recordings", "a5c_as_a4c", 0, 1.0), ("Label changes per recording", "recordings", "cuts_per_recording", 3, 1.0),
    )


    def stream_difference_rows():
        s = temperature["streams"]
        rows = ["| Under the temperature minus as published | With the published decoder setting | With the setting by the rule |", "|---|---|---|"]
        for title, part, key, digits, scale in NUMBERS:
            cells = [signed(s[name]["minus_published"][part][key], digits, scale) for name in (KEPT_SETTING, BY_THE_RULE)]
            rows.append(f"| {title} | " + " | ".join(cells) + " |")
        counts = [s[name]["minus_published"]["recordings"][key] for name in (KEPT_SETTING, BY_THE_RULE) for key in ("a4c_as_a5c", "a5c_as_a4c")]
        assert all(entry["interval"] is None for entry in counts)  # "A count of confused recordings has no interval"
        return rows


    def temperature_text():
        curve = {c["temperature"]: c["log_loss"] for c in temperature["calibration_curve"]}
        before = {c["temperature"]: c["log_loss"] for c in part1["temperature"]["calibration_curve"]}
        assert all(curve[t] == before[t] for t in before) and len(curve) == 40  # the 35 values of part 1 come out again
        assert temperature["chosen"] == 0.05 == min(curve) and temperature["is_the_lowest_tried"] is True and temperature["word"] == "candidate"
        values = [curve[t] for t in sorted(curve)]
        assert all(a < b for a, b in zip(values, values[1:]))  # the log loss on the calibration half rises with every step of the temperature
        as_published, with_it = temperature["selection"]["as published"], temperature["selection"]["with the temperature"]
        assert as_published["errors"] == with_it["errors"] == 1 and with_it["temperature"] == 0.05
        apart = temperature["with_minus_published"]
        at_030 = part1["temperature"]["selection"]["with the temperature"]
        assert at_030["temperature"] == 0.3
        s = temperature["streams"]
        setting, grid = s["setting_by_the_rule"], s["grid_of_the_rule"]
        assert setting["threshold"] == 0.0 and setting["floors_dropped"] is False and (setting["half_window"], setting["threshold_quantile"]) == (5, 0.8)
        kept, ruled = s[KEPT_SETTING]["minus_published"]["multi"]["segment_f1_50"], s[BY_THE_RULE]["minus_published"]["multi"]["segment_f1_50"]
        assert kept["interval"][1] < 0 and ruled["interval"][1] < 0
        none = part1["decoders"]["others"]["no decoder"]["selection"]["multi"]["segment_f1_50"]["value"]
        with_rule = s[BY_THE_RULE]["selection"]["multi"]["segment_f1_50"]["value"]
        assert abs(with_rule - none) < 0.002
        extra = (s["published"]["multi"]["extra_segments"]["value"], s[KEPT_SETTING]["selection"]["multi"]["extra_segments"]["value"])
        return [
            "- **The fit takes 0.05, the lowest value tried again.** On the calibration half the log loss falls with every step down, from "
            f"{rounded(curve[0.30], 4)} at 0.30 to "
            f"{rounded(curve[0.05], 4)} at 0.05. The fit found no best value, and by the plan no lower value is tried.",
            f"- **The word is \"candidate\", as in part 1, and it says little.** With a temperature of 0.05 the log loss of the selection half is "
            f"{f3(with_it['log_loss']['value'])} ({signed(apart['log_loss'])} against as published), where 0.30 gave "
            f"{f3(at_030['log_loss']['value'])}. The same "
            f"difference for the Brier score is {signed(apart['brier'])}. One recording of 284 is wrong either way.",
            "- **At 0.05 the probability is close to a count of votes.** A logit is multiplied by 20 before the probabilities are formed, so a "
            "window gives its "
            "likeliest view a probability near 1 unless two views are almost level. The probability of a recording is then close to the share of its "
            "windows and "
            f"seeds that name the view. The median probability is {rounded(with_it['median_probability'], 3)}. The mean probability is "
            f"{f3(-with_it['probability_minus_right']['value'])} below the share of right recordings, where the published one is "
            f"{f3(-as_published['probability_minus_right']['value'])} below.",
            f"- **The streams lose under the temperature, with either decoder setting.** With the published setting segment F1 on the multi streams is "
            f"{f3(s[KEPT_SETTING]['selection']['multi']['segment_f1_50']['value'])} ({signed(kept)}), and the segments that match nothing rise from "
            f"{rounded(extra[0], 0)} to "
            f"{rounded(extra[1], 0)}. With the rule's own setting it is {f3(with_rule)} ({signed(ruled)}).",
            f"- **Sharp probabilities make thresholds of the rule's grid fall together.** Of its 40 thresholds {grid['thresholds_at_zero']} are "
            "zero, and the 40 hold "
            f"{grid['distinct_thresholds']} distinct values. {setting['n_feasible']} of the 40 settings keep the floors, and the rule takes half window "
            f"{setting['half_window']} with a threshold of zero. With either setting the result is about that of no decoder at all in part 1: "
            f"{f3(s[KEPT_SETTING]['selection']['multi']['segment_f1_50']['value'])} and {f3(with_rule)} against {f3(none)}.",
            "- **What this answers.** At the fitted temperature the decoder loses what it works with. The streams were scored at this one "
            "temperature only. What a "
            "milder one does to them, such as the 0.30 of part 1, was not measured.",
        ]


    # --------------------------------------------------------------------------------------------- the whole section
    def section():
        view_bullets, _ = view_text()
        lines = [
            "## 11. Results of part 3",
            "",
            "**Run on 2026-11-26 from 22:24 to 22:27 UTC with the code of commit `7892e4d`, which holds the plan of section 10.** The output of the command is "
            "[`ablations/open.json`](ablations/open.json). The stored logits are those of part 1 and gave the published number of the selection "
            "half, 0.9594. The 40 "
            "settings and the 35 temperatures of part 1 came out again to the last digit. Intervals are at 95%.",
            "",
            "**Thresholds above the grid.** The 0.999 quantile of the grid and the three added quantiles, for every half window.",
            "",
            *added_rows(),
            "",
            *higher_text(),
            "",
            "**The halves exchanged, in full.** The 40 settings with thresholds from the selection half, where the rule chooses in this exchange.",
            "",
            *exchanged_rows(),
            "",
            *exchanged_text(),
            "",
            "**What the view test drops,** at the published setting.",
            "",
            *view_rows(),
            "",
            *view_bullets,
            "",
            "**The other decoders at every value of their number.**",
            "",
            *other_rows(),
            "",
            "On the selection half, around the value that the rule chose:",
            "",
            *summary_rows(),
            "",
            *other_text(),
            "",
            "**Lower temperatures.**",
            "",
            *curve_rows(),
            "",
            *temperature_rows(),
            "",
            "The streams of the selection half under the fitted temperature:",
            "",
            *stream_rows(),
            "",
            "Every number of section 5, under the temperature minus as published, with the same draws. A count of confused recordings has no interval.",
            "",
            *stream_difference_rows(),
            "",
            *temperature_text(),
            "",
            "**Against what was expected** (section 10).",
            "",
            "- **Higher thresholds: wrong in one point.** An added setting above the published one was expected on the selection half. There is "
            "none. The rule keeps "
            "the published setting and the word is not \"candidate\", both as expected.",
            "- **The halves exchanged: as expected.**",
            "- **The view test: wrong in one point.** Fewer than 10 dropped peaks were expected on each half. There are 14 and 19. No sample changes "
            "on the selection "
            "half, as expected.",
            "- **The other decoders: wrong.** At least 4 of the Viterbi path's 8 costs were expected within 0.02 of the chosen one. There are 2.",
            "- **Lower temperatures: wrong in one point.** With the rule's own setting the streams were expected to lose 0.01 at most. They lose "
            "0.053. The fitted "
            "value, the flat log loss and the loss with the published setting came out as expected.",
            "",
            "**What follows.** By section 6 nothing changes in the published system. Part 3 names no candidate for the decoder. For `classify`'s "
            "probability the "
            "word is \"candidate\" for a temperature of 0.30 in part 1 and for 0.05 here, by the same rule, and the fit has no best value.",
            "",
        ]
        total = view_text()[1]
        assert (total["calibration"]["dropped"], total["selection"]["dropped"]) == (14, 19)
        curve = {c["temperature"]: c["log_loss"] for c in temperature["calibration_curve"]}
        assert curve[0.30] - curve[0.05] < 0.002  # "the flat log loss"
        ruled = temperature["streams"][BY_THE_RULE]["minus_published"]["multi"]["segment_f1_50"]["value"]
        assert f3(-ruled) == "0.053"
        assert ties == []  # of the numbers that part 3 prints, the rule of the ties moves none
        return "\n".join(lines)

    return section()


def main(argv: list[str] | None = None) -> int:
    if not __debug__:
        raise SystemExit("the checks of this tool are assertions: run it without -O")
    args = sys.argv[1:] if argv is None else argv
    files = [json.loads(Path(name).read_text()) for name in args[1:]]
    if args[:1] == ["part2"] and len(files) == 1:
        sys.stdout.write(part_two(*files))
    elif args[:1] == ["part3"] and len(files) == 2:
        sys.stdout.write(part_three(*files))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
