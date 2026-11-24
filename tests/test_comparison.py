"""The comparison with other approaches: counted items, units, numbers, intervals, words and the commands."""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pytest

from echoview import windows
from echoview.evaluation import benchmark, calibrate, comparison, metrics, scorecard
from echoview.evaluation.comparison import ROUTER, Row

RECORDINGS = 12  # rec0 .. rec11, view k % 5, examination k // 3, 40 samples each


def fragment(k: int, start: int = 0, end: int = 40) -> benchmark.Fragment:
    return benchmark.Fragment(f"rec{k}", k % 5, 1.0, tuple(range(start, end)))


def sequences() -> tuple[benchmark.Sequence, ...]:
    """12 single recordings from 4 examinations, 6 pair streams and 8 multi streams built from them."""
    natives = [benchmark.Sequence(f"native_{k}", "native", (fragment(k),), group=k // 3) for k in range(RECORDINGS)]
    pairs = [benchmark.Sequence(f"pairs_{k}", "pairs", (fragment(k, 0, 20), fragment((k + 1) % RECORDINGS, 5, 30))) for k in range(6)]
    multi = [benchmark.Sequence(f"multi_{k}", "multi", tuple(fragment((k + 2 * j) % RECORDINGS, 3 * j, 3 * j + 12 + 4 * j) for j in range(4)))
             for k in range(8)]
    return (*natives, *pairs, *multi)


def truth_labels(seqs: tuple[benchmark.Sequence, ...]) -> dict[str, np.ndarray]:
    return {s.id: s.labels.copy() for s in seqs}


def disturbed(labels: np.ndarray, rng: np.random.Generator, flips: int = 2) -> np.ndarray:
    """The labels with a few runs of 2 to 9 samples given another view."""
    out = labels.copy()
    for _ in range(flips):
        start = int(rng.integers(0, max(1, len(out) - 9)))
        out[start:start + int(rng.integers(2, 10))] = int(rng.integers(0, 5))
    return out


def logits_from(seqs: tuple[benchmark.Sequence, ...], rng: np.random.Generator, noise: float = 0.3) -> dict[str, np.ndarray]:
    """Window logits (windows, 3 seeds, 5) that name the reference view at every window centre."""
    out = {}
    for s in seqs:
        centres = windows.window_centres(len(s))
        out[s.id] = 8.0 * np.eye(5)[s.labels[centres]][:, None, :] + noise * rng.normal(size=(len(centres), 3, 5))
    return out


def test_the_counted_items_give_the_counts_of_the_package():
    rng = np.random.default_rng(7)
    for trial in range(60):
        pieces = [(int(rng.integers(0, 5)), int(rng.integers(8, 40))) for _ in range(int(rng.integers(2, 7)))]
        fragments = tuple(benchmark.Fragment(f"r{trial}_{k}", view, 1.0, tuple(range(length))) for k, (view, length) in enumerate(pieces))
        sequence = benchmark.Sequence("s", "multi", fragments)
        truth, piece, pred = sequence.labels, sequence.pieces, disturbed(sequence.labels, rng, flips=int(rng.integers(0, 5)))
        total: Counter = Counter()
        for column, sample, amount in comparison.stream_items(truth, piece, pred, sequence.joins):
            assert 0 <= sample < len(truth)
            total[column] += amount
        counts = metrics.stream_counts(sequence, pred, 5.0)
        assert (total["seg_matched"], total["seg_pred"], total["seg_true"]) == (counts["seg_matched"], counts["seg_pred"], counts["seg_true"])
        assert (total["cut_hit_5"], total["cut_pred"], total["change"]) == (counts["t_matched"], counts["p_pred"], counts["t_true"])
        assert (total["same_join"], total["false_split"]) == (counts["same_join"], counts["false_split"])
        assert total["extra_short"] + total["extra_long"] == counts["seg_pred"] - counts["seg_matched"]
        assert total["missed_short"] + total["missed_long"] == counts["seg_true"] - counts["seg_matched"]
        assert sum(total[f"seg_matched_v{v}"] for v in range(5)) == counts["seg_matched"]
        for overlap in comparison.OVERLAPS:
            assert (total[f"tp{overlap}"], total[f"fp{overlap}"], total[f"fn{overlap}"]) == scorecard.overlap_counts(truth, pred, overlap / 100)
        card = scorecard.stream_counts(truth, piece, pred)
        assert {kind: total[kind] for kind in scorecard.CATEGORIES} == card["split"]
        assert (total["piece"], total["piece_right"]) == (card["pieces"], np.trace(card["piece_midpoints"]))
        assert (total["segment"], total["segment_right"]) == (card["segments"], np.trace(card["segment_midpoints"]))
        assert total["piece_short"] + total["piece_long"] == card["pieces"] and total["floor_wrong"] == card["floor_wrong"]
        assert all(total[f"conf_{t}_{p}"] == card["frame"][t, p] for t in range(5) for p in range(5))
        offsets, _ = scorecard.cut_offsets(truth, pred)
        assert total["near_matched"] == len(offsets) and total["near_late"] == sum(o > 0 for o in offsets)
        assert total["near_abs"] == pytest.approx(sum(abs(o) for o in offsets) / 10)
        assert all(total[f"cut_hit_{a}"] <= total[f"cut_hit_{b}"] for a, b in zip(comparison.TOLERANCES, comparison.TOLERANCES[1:]))


def test_a_row_without_a_view_for_every_sample_is_refused():
    with pytest.raises(ValueError, match="needs one of the views"):
        comparison.stream_items(np.zeros(10, int), np.zeros(10, int), np.full(10, -1), np.array([], int))


def test_every_item_is_counted_once_in_each_unit():
    seqs, rng = sequences(), np.random.default_rng(3)
    labels = {s.id: disturbed(s.labels, rng) for s in seqs}
    examination = {s.fragments[0].video: s.group for s in seqs if s.kind == "native"}
    multi = benchmark.of_kind(seqs, "multi")
    tables = comparison.stream_tables(multi, labels, examination)
    assert tables["streams"].shape[0] == 8 and tables["examinations"].shape[0] == 4
    for name, k in comparison.STREAM_INDEX.items():
        if name in ("edit", "streams"):
            assert tables["examinations"][:, k].sum() == 0  # the Edit score exists per stream only
        else:
            assert tables["examinations"][:, k].sum() == pytest.approx(tables["streams"][:, k].sum())
    assert (tables["examinations"].sum(1) > 0).all()  # every examination holds some of the items
    assert set(comparison.stream_tables(multi, labels, None)) == {"streams"}
    natives = benchmark.of_kind(seqs, "native")
    assert comparison.native_tables(natives, labels)["examinations"].shape[0] == 4
    ungrouped = tuple(benchmark.Sequence(s.id, s.kind, s.fragments) for s in natives)
    assert comparison.native_tables(ungrouped, labels)["recordings"].shape[0] == RECORDINGS


def test_the_numbers_are_those_of_the_package():
    seqs, rng = sequences(), np.random.default_rng(11)
    labels = {s.id: disturbed(s.labels, rng, flips=3) for s in seqs}
    rows = [Row(ROUTER, ROUTER, "A", labels), Row(f"{ROUTER}, no decoder", ROUTER, "N", truth_labels(seqs))]
    result = comparison.compare(seqs, rows, draws=200, seed=1)
    ours, package = result["rows"][ROUTER], metrics.evaluate(labels, seqs)
    assert ours["multi"]["segment_f1_50"]["value"] == pytest.approx(package["segment_f1_50_multi"])
    assert ours["pairs"]["segment_f1_50"]["value"] == pytest.approx(package["segment_f1_50_pairs"])
    assert ours["multi"]["cut_recall"]["value"] == pytest.approx(package["boundary_recall_multi"])
    assert ours["multi"]["cut_precision"]["value"] == pytest.approx(package["boundary_precision_multi"])
    assert ours["multi"]["same_view_cut_rate"]["value"] == pytest.approx(package["same_class_cut_rate_multi"])
    assert ours["multi"]["wrong_samples"]["value"] == pytest.approx(1 - package["multi_frame_accuracy"])
    assert ours["recordings"]["accuracy"]["value"] == pytest.approx(package["native_accuracy"])
    assert ours["recordings"]["macro_f1"]["value"] == pytest.approx(package["native_macro_f1"])
    assert ours["recordings"]["cuts_per_recording"]["value"] == pytest.approx(package["native_internal_cuts_per_clip"])
    multi = benchmark.of_kind(seqs, "multi")
    card = scorecard.summarise([(s.labels, s.pieces, labels[s.id]) for s in multi])
    assert ours["multi"]["piece_midpoint"]["value"] == pytest.approx(card["A1_piece"]["accuracy"])
    assert ours["multi"]["segment_midpoint"]["value"] == pytest.approx(card["A1_segment"]["accuracy"])
    assert ours["multi"]["edit"]["value"] == pytest.approx(card["A6_edit"])
    assert ours["multi"]["grid_floor"]["value"] == pytest.approx(card["grid_floor"])
    for overlap in comparison.OVERLAPS:
        assert ours["multi"][f"mstcn_f1_{overlap}"]["value"] == pytest.approx(card["A6_f1"][str(overlap)])
    for v, view in enumerate(("PLAX", "PSAX", "A4C", "A5C", "SC4C")):
        for mine, theirs in ((f"recovery_{view}", card["A5_recovery"][v]), (f"purity_{view}", card["A5_purity"][v])):
            if theirs is not None:
                assert ours["multi"][mine]["value"] == pytest.approx(theirs)
    assert ours["multi"]["cut_matched_share"]["value"] == pytest.approx(card["A4_cuts"]["matched_share"])
    assert ours["multi"]["edit"]["examinations"] is None and ours["multi"]["edit"]["streams"] is not None
    assert ours["multi"]["segment_f1_50"]["interval"] in (ours["multi"]["segment_f1_50"]["streams"], ours["multi"]["segment_f1_50"]["examinations"])
    perfect = result["rows"][f"{ROUTER}, no decoder"]
    assert perfect["multi"]["segment_f1_50"]["value"] == 1.0 and perfect["multi"]["wrong_samples"]["value"] == 0.0
    assert perfect["multi"]["extra_segments"] == {"value": 0.0, "support": 0.0, "streams": None, "examinations": None, "interval": None}
    assert result["best_rows"]["multi"]["segment_f1_50"] == [f"{ROUTER}, no decoder"]
    assert "segments_per_reference" not in result["best_rows"]["multi"]  # no direction, no best row


def test_the_wider_interval_is_the_one_reported():
    assert comparison.wider({"streams": [0.1, 0.2], "examinations": [0.0, 0.3]}) == [0.0, 0.3]
    assert comparison.wider({"streams": [0.1, 0.2], "examinations": None}) == [0.1, 0.2]
    assert comparison.wider({"streams": None}) is None
    assert comparison.percentiles(np.array([np.nan, np.nan]), 95.0) is None
    assert comparison.percentiles(np.arange(101.0), 95.0) == pytest.approx([2.5, 97.5])
    assert comparison.percentiles(np.arange(801.0), 98.75) == pytest.approx([5.0, 795.0])


def misread_second_piece(seqs: tuple[benchmark.Sequence, ...]) -> dict[str, np.ndarray]:
    out = truth_labels(seqs)
    for s in seqs:
        if s.kind == "multi":
            second = s.pieces == 1
            out[s.id][second] = (s.labels[second] + 1) % 5
    return out


def test_the_words_of_the_named_comparisons():
    seqs = sequences()
    right, worse = truth_labels(seqs), misread_second_piece(seqs)
    rows = [Row(ROUTER, ROUTER, "A", right), Row(f"{ROUTER} seed 0", ROUTER, "S", worse),
            Row("EchoViewCLIP", "EchoViewCLIP", "A", worse), Row("EchoViewCLIP + decoder", "EchoViewCLIP", "D", worse),
            Row("STFM", "STFM", "A", worse), Row("STFM + decoder", "STFM", "D", right),
            Row("ASFormer", "ASFormer", "A", worse), Row("ASFormer + released numbers", "ASFormer", "C", worse),
            Row("EchoPrime per frame", "EchoPrime", "A", worse), Row("EchoPrime + decoder", "EchoPrime", "D", right)]
    scored = comparison.score_rows(rows, seqs, draws=300, seed=5)
    named = {entry["system"]: entry for entry in comparison.named_comparisons(rows, scored)}
    assert named["EchoViewCLIP"]["word"] == "ahead of" and named["EchoViewCLIP"]["named"]["level"] == 98.75
    assert named["EchoViewCLIP"]["named"]["interval"][0] > 0 and named["EchoViewCLIP"]["decoder_adds"]["value"] == 0.0
    assert named["STFM"]["word"] == "not shown to differ from" and named["STFM"]["named"]["interval"] == [0.0, 0.0]
    assert named["STFM"]["decoder_adds"]["value"] > 0 and named["STFM"]["remains"]["value"] == 0.0
    assert named["STFM"]["published_gap"]["value"] == pytest.approx(named["STFM"]["remains"]["value"] + named["STFM"]["decoder_adds"]["value"])
    assert named["ASFormer"]["word"] == "not answered" and named["ASFormer"]["row_with_decoder"] is None  # row C never stands in
    assert named["EchoPrime"]["row_as_published"] == named["EchoPrime"]["row_with_decoder"] == "EchoPrime + decoder"
    behind = {entry["system"]: entry["word"] for entry in comparison.named_comparisons(rows, scored, f"{ROUTER} seed 0")}
    assert behind == {"EchoViewCLIP": "not shown to differ from", "STFM": "behind", "ASFormer": "not answered", "EchoPrime": "behind"}
    assert comparison.word(None) == "not answered" and comparison.word([0.01, 0.05]) == "ahead of"
    assert comparison.word([-0.01, 0.05]) == "not shown to differ from" and comparison.word([-0.05, -0.01]) == "behind"


def test_ahead_needs_the_published_row_too():
    """Ahead of the decoder row but not of the published row: the word stays "not shown to differ from"."""
    seqs = sequences()
    right, worse = truth_labels(seqs), misread_second_piece(seqs)
    rows = [Row(ROUTER, ROUTER, "A", right), Row("STFM", "STFM", "A", right), Row("STFM + decoder", "STFM", "D", worse)]
    entry = comparison.named_comparisons(rows, comparison.score_rows(rows, seqs, draws=300, seed=5))[0]
    assert entry["named"]["interval"][0] > 0 and entry["published_gap"]["interval"] == [0.0, 0.0] and entry["word"] == "not shown to differ from"


def stores_for(seqs: tuple[benchmark.Sequence, ...], rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    """Stored rows of two other systems: one sharp, and one that flickers and reads SC4C as PLAX, so that no decoder
    setting meets the floor for single recordings. Rows sum to less than 1, as some stores do."""
    sharp = {s.id: 0.9 * np.eye(5)[s.labels] + 0.01 for s in seqs}
    flicker = {s.id: 0.5 * np.eye(5)[disturbed(np.where(s.labels == 4, 0, s.labels), rng, flips=3)] + 0.08 * rng.random((len(s), 5)) for s in seqs}
    return {"EchoViewCLIP": sharp, "EchoPrime": flicker}


def test_rows_follow_the_plan():
    seqs, rng = sequences(), np.random.default_rng(2)
    logits, stores = logits_from(seqs, rng), stores_for(seqs, rng)
    setting = {"half_window": 5, "threshold": 0.2, "nms_radius": 5, "threshold_quantile": 0.9}
    chosen = {"seeds": {"0": {"params": setting}, "2": {"params": setting}}, "systems": {"EchoViewCLIP": {"params": setting}}}
    rows = comparison.build_rows(seqs, logits, stores, chosen)
    roles = {row.name: row.role for row in rows}
    assert roles == {ROUTER: "A", f"{ROUTER}, no decoder": "N", "Perfect on the 0.5 s grid": "P", f"{ROUTER} seed 0": "S", f"{ROUTER} seed 2": "S",
                     "EchoViewCLIP": "A", "EchoViewCLIP + decoder": "D", "EchoViewCLIP + released numbers": "C",
                     "EchoPrime per frame": "A", "EchoPrime + decoder": "D", "EchoPrime + released numbers": "C"}
    by_name = {row.name: row for row in rows}
    assert all(np.array_equal(by_name["EchoViewCLIP"].labels[s.id], s.labels) for s in seqs)  # the likeliest view of the stored rows
    assert all(set(row.labels) == {s.id for s in seqs} for row in rows)
    without = comparison.build_rows(seqs, logits, {"STFM": stores["EchoPrime"]}, {})
    assert [row.name for row in without] == [ROUTER, f"{ROUTER}, no decoder", "Perfect on the 0.5 s grid", "STFM", "STFM + released numbers"]
    result = comparison.compare(seqs, rows, draws=100, seed=1)
    assert set(result["seeds"]) == {f"{ROUTER} seed 0", f"{ROUTER} seed 2"} and set(result["decoder_on_echorouter"]) == {"multi", "pairs"}
    only = result["differences_from_echorouter"]["EchoPrime per frame"]["recordings_only_one_reads_right"]
    assert set(only) == {"only_first_right", "only_second_right"}


def test_the_check_of_the_published_numbers_stops_before_scoring(monkeypatch):
    seqs = sequences()
    rows = [Row(ROUTER, ROUTER, "A", truth_labels(seqs))]
    perfect = {"segment_f1_50_multi": 1.0, "segment_f1_50_pairs": 1.0, "native_accuracy": 1.0, "native_macro_f1": 1.0}
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: perfect})
    assert comparison.published_check(rows, seqs) == {ROUTER: perfect}
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: {**perfect, "native_accuracy": 0.999999}})
    with pytest.raises(SystemExit, match="nothing was scored"):
        comparison.published_check(rows, seqs)
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: perfect, "STFM": perfect})
    with pytest.raises(SystemExit, match="Rows without outputs: \\['STFM'\\]"):
        comparison.published_check(rows, seqs)


def write_store(folder, seqs, rows, names):
    folder.mkdir(parents=True)
    for s in seqs:
        np.savez(comparison.store_file(folder, s, names), prob=rows[s.id].astype(np.float32))


def test_a_store_is_read_by_folder_name_and_stream_id(tmp_path):
    seqs, rng = sequences(), np.random.default_rng(4)
    names = {f"rec{k}": f"folder of rec{k}" for k in range(RECORDINGS)}
    stored = stores_for(seqs, rng)["EchoViewCLIP"]
    write_store(tmp_path / "store", seqs, stored, names)
    assert (tmp_path / "store" / "folder of rec3.npz").is_file() and (tmp_path / "store" / "multi_0.npz").is_file()
    rows, digest = comparison.read_store(tmp_path / "store", seqs, names)
    assert all(np.allclose(rows[s.id], stored[s.id]) for s in seqs) and len(digest) == 64
    assert comparison.read_store(tmp_path / "store", seqs, names)[1] == digest
    held = comparison.coverage(tmp_path / "store", seqs, names)
    assert (held["stored"], held["sequences"], held["wrong_shape"], held["arrays"], held["digest"]) == (len(seqs), len(seqs), 0, ["prob"], digest)
    assert held["row_sums"][1] < 1.0
    (tmp_path / "store" / "multi_0.npz").unlink()
    assert comparison.coverage(tmp_path / "store", seqs, names) == {"sequences": len(seqs), "stored": len(seqs) - 1}
    with pytest.raises(SystemExit, match="no stored output"):
        comparison.read_store(tmp_path / "store", seqs, names)
    np.savez(tmp_path / "store" / "multi_0.npz", prob=np.zeros((3, 5)))
    with pytest.raises(SystemExit, match="wrong shape"):
        comparison.read_store(tmp_path / "store", seqs, names)


def test_identical_rows_differ_by_nothing_in_every_draw():
    """The pairing: two rows with the same wrong labels share every resample, so every interval of their difference is
    exactly zero. Rows that were resampled apart would differ."""
    seqs, rng = sequences(), np.random.default_rng(9)
    labels = {s.id: disturbed(s.labels, rng, flips=3) for s in seqs}
    rows = [Row(ROUTER, ROUTER, "A", labels), Row(f"{ROUTER}, no decoder", ROUTER, "N", {k: v.copy() for k, v in labels.items()})]
    result = comparison.compare(seqs, rows, draws=200, seed=1)
    seen = 0
    for part, numbers in result["differences_from_echorouter"][f"{ROUTER}, no decoder"].items():
        if part == "recordings_only_one_reads_right":
            assert numbers == {"only_first_right": 0, "only_second_right": 0}
            continue
        for name, entry in numbers.items():
            assert entry["value"] in (0.0, None), name
            for unit in ("streams", "examinations"):
                if entry.get(unit) is not None:
                    assert entry[unit] == [0.0, 0.0], (part, name, unit)
                    seen += 1
    assert seen > 100
    assert result["rows"][ROUTER]["multi"]["segment_f1_50"]["value"] < 1.0  # the rows are wrong somewhere, or the test says nothing


def test_an_item_goes_to_the_examination_at_its_sample():
    """A stream of three pieces: rec0 (examination 0, view 0, 20 samples), rec3 (examination 1, view 3, 10 samples) and
    rec8 (examination 2, view 3, 30 samples). The row puts the cut 2 samples late and reads everything else right."""
    stream = benchmark.Sequence("multi_x", "multi", (fragment(0, 0, 20), fragment(3, 0, 10), fragment(8, 0, 30)))
    examination = {"rec0": 0, "rec3": 1, "rec8": 2}
    pred = np.array([0] * 22 + [3] * 38)
    tables = comparison.stream_tables((stream,), {"multi_x": pred}, examination)
    by = {name: tables["examinations"][:, k].tolist() for name, k in comparison.STREAM_INDEX.items()}
    assert by["change"] == [0, 1, 0] and by["cut_hit_5"] == [0, 1, 0] and by["cut_hit_1"] == [0, 0, 0]  # counted at the view change, sample 20
    assert by["cut_pred"] == [0, 1, 0] and by["near_abs"] == [0, 0.2, 0] and by["near_late"] == [0, 1, 0]  # the cut is at sample 22
    assert by["same_join"] == [0, 0, 1] and by["false_split"] == [0, 0, 0]  # the join of two pieces of view 3 is at sample 30
    assert by["piece"] == [1, 1, 1] and by["piece_right"] == [1, 1, 1] and by["piece_short"] == [0, 1, 0] and by["piece_long"] == [1, 0, 1]
    assert by["seg_true"] == [1, 0, 1]  # reference segments: 0..20 (middle 9) and 20..60 (middle 39, in rec8)
    assert by["seg_pred"] == [1, 0, 1] and by["seg_matched"] == [1, 0, 1]  # returned: 0..22 (middle 10) and 22..60 (middle 40)
    assert by["conf_0_0"] == [20, 0, 0] and by["conf_3_0"] == [0, 2, 0] and by["conf_3_3"] == [0, 8, 30]
    assert by["edge"] == [0, 2, 0] and by["misread"] == [0, 0, 0]
    assert tables["streams"][0, comparison.STREAM_INDEX["edit"]] == 100.0


def test_short_and_long_are_told_apart():
    """Pieces of 10 and 30 samples; the short one is read as another view."""
    stream = benchmark.Sequence("multi_y", "multi", (fragment(0, 0, 30), fragment(3, 0, 10), fragment(5, 0, 30)))
    pred = np.array([0] * 30 + [4] * 10 + [0] * 30)
    rows = [Row(ROUTER, ROUTER, "A", {"multi_y": pred})]
    scored = comparison.score_tables(comparison.STREAM_STATS, comparison.stream_tables((stream,), rows[0].labels, None), comparison.STREAM_INDEX,
                                     {"streams": np.ones((3, 1))})
    values = scored.values
    assert (values["piece_midpoint_short"], values["piece_midpoint_long"], values["piece_midpoint"]) == (0.0, 1.0, pytest.approx(2 / 3))
    assert (values["missed_segments"], values["missed_segments_short"], values["extra_segments"], values["extra_segments_short"]) == (1.0, 1.0, 1.0, 1.0)
    assert values["wrong_misread"] == pytest.approx(10 / 70) and values["wrong_edge"] == 0.0 and values["segments_per_reference"] == 1.0
    assert values["piece_midpoint_A5C"] == 0.0 and values["piece_midpoint_PLAX"] == 1.0 and values["recovery_A5C"] == 0.0


def test_a_count_of_five_or_fewer_has_no_interval():
    for count, has_interval in ((5, False), (6, True)):
        scored = comparison.Scored({"n": float(count)}, {"n": float(count)}, {"streams": {"n": np.arange(50.0)}})
        assert (comparison.report(scored)["n"]["interval"] is not None) == has_interval
        assert (comparison.difference(scored, scored, "n", 95.0)["interval"] is not None) == has_interval
    assert comparison.percentiles(np.array([1.0, 2.0, np.nan, np.inf, 3.0]), 95.0) == pytest.approx([1.05, 2.95])
    assert comparison.hull({"streams": [-0.001, 0.01], "examinations": [0.005, 0.2]}) == [-0.001, 0.2]
    assert comparison.hull({"streams": None}) is None
    assert comparison._clean({"a": np.float64("nan"), "b": [np.int64(2), float("inf")], "c": np.bool_(True)}) == {"a": None, "b": [2, None], "c": True}


def test_the_word_needs_every_unit():
    """Ahead by examinations but not by streams: the hull holds zero, so the word is not "ahead of"."""
    def scored(by_streams: float, by_examinations: float) -> comparison.Scored:
        return comparison.Scored({"segment_f1_50": 0.9}, {"segment_f1_50": 100.0},
                                 {"streams": {"segment_f1_50": np.linspace(by_streams - 0.01, by_streams + 0.01, 400)},
                                  "examinations": {"segment_f1_50": np.linspace(by_examinations - 0.05, by_examinations + 0.05, 400)}})

    zero = comparison.Scored({"segment_f1_50": 0.7}, {"segment_f1_50": 100.0}, {unit: {"segment_f1_50": np.zeros(400)} for unit in ("streams", "examinations")})
    rows = [Row(ROUTER, ROUTER, "A", {}), Row("STFM", "STFM", "A", {}), Row("STFM + decoder", "STFM", "D", {})]
    entry = comparison.named_comparisons(rows, {ROUTER: {"multi": scored(0.005, 0.2)}, "STFM": {"multi": zero}, "STFM + decoder": {"multi": zero}})[0]
    assert entry["named"]["examinations"][0] > 0 and entry["named"]["streams"][0] < 0 and entry["word"] == "not shown to differ from"
    clear = comparison.named_comparisons(rows, {ROUTER: {"multi": scored(0.2, 0.2)}, "STFM": {"multi": zero}, "STFM + decoder": {"multi": zero}})[0]
    assert clear["word"] == "ahead of" and clear["named"]["hull"][0] > 0


def test_best_rows_and_recordings_only_one_reads_right():
    seqs = sequences()
    right, worse = truth_labels(seqs), misread_second_piece(seqs)
    wrong_natives = {**right, "native_0": (right["native_0"] + 1) % 5, "native_1": (right["native_1"] + 2) % 5}
    rows = [Row(ROUTER, ROUTER, "A", wrong_natives), Row(f"{ROUTER}, no decoder", ROUTER, "N", worse), Row("Perfect on the 0.5 s grid", "", "P", right)]
    result = comparison.compare(seqs, rows, draws=50, seed=1)
    assert result["best_rows"]["multi"]["wrong_samples"] == [ROUTER] and result["best_rows"]["multi"]["missed_segments"] == [ROUTER]  # lower is better
    assert result["best_rows"]["recordings"]["accuracy"] == [f"{ROUTER}, no decoder"]  # row P is never a best row
    only = result["differences_from_echorouter"][f"{ROUTER}, no decoder"]["recordings_only_one_reads_right"]
    assert only == {"only_first_right": 0, "only_second_right": 2}


def test_the_commands_choose_score_and_write_once(tmp_path, monkeypatch, capsys):
    seqs, rng = sequences(), np.random.default_rng(6)
    names = {f"rec{k}": f"rec{k}" for k in range(RECORDINGS)}
    ungrouped = tuple(benchmark.Sequence(s.id, s.kind, s.fragments) for s in seqs)  # the development benchmark names no examination
    monkeypatch.setattr(comparison.benchmark, "load", lambda: {"calibration": ungrouped, "selection": ungrouped})
    monkeypatch.setattr(comparison.benchmark, "load_test", lambda: seqs)
    monkeypatch.setattr(comparison, "_names", lambda images, sequences: names)
    monkeypatch.setattr(comparison, "PUBLISHED", {})
    logits, two = logits_from(seqs, rng), stores_for(seqs, rng)
    stores = {"EchoViewCLIP": two["EchoViewCLIP"], "STFM": two["EchoPrime"], "ASFormer": two["EchoViewCLIP"], "EchoPrime": two["EchoPrime"]}
    np.savez(tmp_path / "logits.npz", **logits)
    for name, rows in stores.items():
        write_store(tmp_path / name, seqs, rows, names)
    ensemble = comparison.choose(comparison.member_probabilities(logits, seqs, (0, 1, 2)), seqs, drop_floors=False)
    assert ensemble["status"] == "feasible"
    monkeypatch.setattr(comparison, "B3", {k: ensemble["params"][k] for k in ("half_window", "threshold", "nms_radius")})
    monkeypatch.setattr(comparison, "RELEASED_QUANTILE", ensemble["params"]["threshold_quantile"])
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    common = ["--images", str(tmp_path), "--logits", str(tmp_path / "logits.npz")]
    four = [arg for name in stores for arg in ("--store", f"{name}={tmp_path / name}")]
    assert comparison.main(["inventory", "--images", str(tmp_path), *four, "--out", str(tmp_path / "inventory.json")]) == 0
    listed = json.loads((tmp_path / "inventory.json").read_text())
    assert listed["EchoViewCLIP"]["files"] == len(seqs) and listed["EchoViewCLIP"]["test"]["stored"] == len(seqs)
    development = ["--store", f"EchoViewCLIP={tmp_path / 'EchoViewCLIP'}", "--store", f"STFM={tmp_path / 'STFM'}"]
    with pytest.raises(SystemExit, match="give no store for it here"):
        comparison.main(["settings", *common, "--store", f"EchoPrime={tmp_path / 'EchoPrime'}", "--out", str(tmp_path / "no.json")])
    assert comparison.main(["settings", *common, *development, "--out", str(tmp_path / "settings.json")]) == 0
    chosen = json.loads((tmp_path / "settings.json").read_text())
    assert set(chosen["seeds"]) == {"0", "1", "2"} and set(chosen["systems"]) == {"EchoViewCLIP", "STFM"}
    assert chosen["without_development_outputs"] == ["ASFormer"]
    assert chosen["systems"]["EchoViewCLIP"]["status"] == "feasible" and not chosen["systems"]["EchoViewCLIP"]["floors_dropped"]
    assert chosen["systems"]["STFM"]["floors_dropped"] and chosen["systems"]["STFM"]["n_feasible"] == 0
    as_stored = comparison.read_store(tmp_path / "STFM", ungrouped, names)[0]
    unconstrained = calibrate.choose(as_stored, ungrouped)
    assert chosen["systems"]["STFM"]["params"] == unconstrained["best_unconstrained"]["params"] != unconstrained["params"]  # not the diagnostic setting
    assert len(chosen["stores"]["STFM"]["digest"]) == 64
    out = tmp_path / "out" / "comparison.json"

    def score(systems=tuple(stores), chosen_file="settings.json", draws="100") -> list[str]:
        given = [arg for name in systems for arg in ("--store", f"{name}={tmp_path / name}")]
        return ["score", "--benchmark", "test", *common, *given, "--settings", str(tmp_path / chosen_file), "--draws", draws, "--out", str(out)]

    monkeypatch.setattr(comparison, "DRAWS", 100)
    with pytest.raises(SystemExit, match="the plan fixes 100 draws"):
        comparison.main(score(draws="50"))
    with pytest.raises(SystemExit, match="all four stores are needed"):
        comparison.main(score(systems=("EchoViewCLIP", "STFM", "EchoPrime")))
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the one the plan names"):
        comparison.main(score())
    monkeypatch.setattr(comparison, "TEST_LOGITS_SHA256", comparison.file_sha256(tmp_path / "logits.npz"))
    (tmp_path / "partial.json").write_text(json.dumps({**chosen, "seeds": {"0": chosen["seeds"]["0"]}, "without_development_outputs": []}))
    with pytest.raises(SystemExit, match=r"each of the three seeds; .*either its setting or that it has no development outputs"):
        comparison.main(score(chosen_file="partial.json"))
    monkeypatch.setattr(comparison, "PUBLISHED", {ROUTER: {"segment_f1_50_multi": 0.5}})
    with pytest.raises(SystemExit, match="nothing was scored"):
        comparison.main(score())
    assert not out.exists()  # a stopped read leaves no output
    monkeypatch.setattr(comparison, "PUBLISHED", {})
    assert comparison.main(score()) == 0 and "wrote" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["benchmark"] == "test" and result["draws"] == 100 and set(result["sha256"]) == {"logits", "settings", *stores}
    named = {entry["system"]: entry for entry in result["named_comparisons"]}
    assert set(named) == set(stores) and named["ASFormer"]["word"] == "not answered" and named["EchoPrime"]["row_without_decoder"] == "EchoPrime per frame"
    assert "examinations" in result["rows"][ROUTER]["multi"]["segment_f1_50"]
    assert not re.search(r"rec\d|native_\d|multi_\d|pairs_\d", json.dumps(result))  # pooled numbers only: no recording or stream is named
    assert "NaN" not in out.read_text() and "Infinity" not in out.read_text()
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="does not replace"):
        comparison.main(score())
    assert out.read_bytes() == before
    with pytest.raises(SystemExit, match="NAME=FOLDER"):
        comparison.main(["inventory", "--images", str(tmp_path), "--store", "Other=x", "--out", str(tmp_path / "x.json")])
    # the selection half, as stage A tries the code: no examinations there, and the logits must give the published number
    trial = ["score", "--benchmark", "selection", *common, *development, "--settings", str(tmp_path / "settings.json"), "--draws", "100"]
    with pytest.raises(SystemExit, match=re.escape("must give 0.9594")):
        comparison.main([*trial, "--out", str(tmp_path / "selection.json")])
    f1 = metrics.evaluate(comparison.decoded(comparison.member_probabilities(logits, seqs, (0, 1, 2)), comparison.B3), seqs)["segment_f1_50_multi"]
    monkeypatch.setattr(comparison, "PUBLISHED_SELECTION_F1", round(f1, 4))
    assert comparison.main([*trial, "--out", str(tmp_path / "selection.json")]) == 0
    tried = json.loads((tmp_path / "selection.json").read_text())
    assert "examinations" not in tried["rows"][ROUTER]["multi"]["segment_f1_50"] and "recordings" in tried["rows"][ROUTER]["recordings"]["accuracy"]
    assert {entry["system"]: entry["word"] for entry in tried["named_comparisons"]}.keys() == {"EchoViewCLIP", "STFM"}
    monkeypatch.setattr(comparison, "B3", {"half_window": 15, "threshold": 0.5, "nms_radius": 5})
    with pytest.raises(SystemExit, match="do not give the released decoder setting"):
        comparison.main(["settings", *common, "--out", str(tmp_path / "other.json")])
