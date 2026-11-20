"""The two-part report of an output with uncertain samples: the labelled part and the uncertain part."""
from __future__ import annotations

import numpy as np
import pytest

from echoview.evaluation import benchmark, metrics, scorecard, selective


def sequence(kind: str, *parts: tuple[str, int, int], name: str = "s") -> benchmark.Sequence:
    """A sequence of fragments (recording, view, length)."""
    fragments = tuple(benchmark.Fragment(video, view, 1.0, tuple(range(length))) for video, view, length in parts)
    return benchmark.Sequence(name, kind, fragments)


@pytest.fixture()
def stream() -> tuple[benchmark.Sequence, np.ndarray, np.ndarray]:
    """Three pieces of 2 s (views 0, 2, 3). The system cuts 0.2 s late and reads the third piece as view 2.
    The band: three samples on each side of the cut at 22."""
    decided = np.array([0] * 22 + [2] * 38)
    band = np.zeros(60, bool)
    band[19:25] = True
    return sequence("multi", ("a", 0, 20), ("b", 2, 20), ("c", 3, 20)), decided, band


def test_the_counts_of_a_stream_with_a_band_around_its_cut(stream):
    seq, decided, band = stream
    c = selective.counts(seq, decided, band, band)
    assert (c["n"], c["labelled"], c["uncertain"]) == (60, 54, 6)
    assert (c["wrong"], c["wrong_labelled"], c["wrong_uncertain"], c["right_uncertain"]) == (22, 20, 2, 4)
    # the middle of every reference piece: a and b right and labelled, c wrong and labelled
    assert (c["pieces"], c["piece_right_kept"], c["piece_right_declined"], c["piece_wrong_kept"], c["piece_wrong_declined"]) == (3, 2, 0, 1, 0)
    assert c["pieces_unlabelled"] == 0
    # the labelled runs the system returns: samples 0-18 (right at its middle) and 25-59 (its middle is in piece c: wrong)
    assert (c["segments_kept"], c["segments_kept_right"]) == (2, 1)
    assert (c["intervals"], c["intervals_no_change"], c["uncertain_near_change"]) == (1, 0, 6)
    assert (c["changes"], c["changes_located"]) == (2, 1)  # the change at 20 has uncertain samples beside it, the one at 40 has none
    assert (c["edge"], c["edge_declined"], c["misread"], c["misread_declined"]) == (2, 2, 20, 0)
    assert (c["spurious"], c["absorbed"]) == (0, 0)
    assert (c["a4c_a5c"], c["a4c_a5c_declined"]) == (20, 0)  # piece c: A5C read as A4C
    assert (c["same_joins"], c["same_joins_uncertain"], c["recordings_right"], c["recordings_wrong"]) == (0, 0, 0, 0)
    assert c["kept_3_2"] == 20 and c["kept_0_0"] == 19 and c["kept_2_2"] == 15 and c["reference_3"] == 20
    assert (c["added"], c["added_wrong"], c["after_band_labelled"], c["after_band_wrong"]) == (0, 0, 54, 20)  # nothing beyond the band


def test_the_ratios_of_the_two_parts(stream):
    seq, decided, band = stream
    r = selective.ratios(selective.counts(seq, decided, band, band))
    assert r["coverage"] == pytest.approx(54 / 60) and r["uncertain_share"] == pytest.approx(6 / 60)
    assert r["error_published"] == pytest.approx(22 / 60) and r["error_labelled"] == pytest.approx(20 / 54)
    assert r["error_labelled"] > r["error_published"]  # declining does not lower the error by itself: this band took 4 right samples for 2 wrong
    assert r["error_ideal"] == pytest.approx(16 / 54)  # six declined, wrong ones first
    assert r["error_if_forced"] == pytest.approx(2 / 6) and r["errors_declined"] == pytest.approx(2 / 22) and r["price"] == pytest.approx(4 / 38)
    assert r["piece_accuracy_labelled"] == pytest.approx(2 / 3) and r["piece_accuracy_published"] == pytest.approx(2 / 3)
    assert r["segment_accuracy"] == 0.5 and r["changes_located"] == 0.5 and r["uncertain_near_change"] == 1.0
    assert r["edge_declined"] == 1.0 and r["misread_declined"] == 0.0 and np.isnan(r["spurious_declined"])
    assert r["purity"][2] == pytest.approx(15 / 35) and r["purity"][0] == 1.0 and np.isnan(r["purity"][1])
    assert r["recovery"][3] == 0.0 and r["recovery"][0] == pytest.approx(19 / 20)
    assert np.isnan(r["error_added"]) and r["error_after_band"] == pytest.approx(20 / 54)


def test_a_test_that_declines_the_wrong_piece_is_seen_beyond_the_band(stream):
    seq, decided, band = stream
    mask = band.copy()
    mask[40:60] = True  # a test for whole pieces declines piece c
    c = selective.counts(seq, decided, mask, band)
    r = selective.ratios(c)
    assert (c["piece_wrong_kept"], c["piece_wrong_declined"], c["pieces_unlabelled"]) == (0, 1, 1)
    assert (c["added"], c["added_wrong"]) == (20, 20) and r["error_added"] == 1.0 and r["error_after_band"] == pytest.approx(20 / 54)
    assert r["error_labelled"] == 0.0 and r["errors_declined"] == 1.0 and r["misread_declined"] == 1.0 and r["a4c_a5c_declined"] == 1.0
    assert c["intervals"] == 2 and c["changes_located"] == 2  # sample 40 is uncertain: the second change now has one beside it
    rows = selective.pieces(seq, decided, mask, band)
    assert [(row["recording"], row["wrong"], row["declined"], row["beyond_band"], row["empty"]) for row in rows] == [
        ("a", False, False, True, False), ("b", False, False, True, False), ("c", True, True, True, True)]


def test_single_recordings_and_joins_of_one_view():
    native = sequence("native", ("a", 1, 40))
    mask = np.zeros(40, bool)
    mask[10:14] = True
    c = selective.counts(native, np.ones(40, int), mask)
    assert (c["recordings_right"], c["recordings_right_uncertain"], c["recordings_wrong"]) == (1, 1, 0)  # a false alarm
    assert (c["intervals"], c["intervals_no_change"], c["uncertain_near_change"]) == (1, 1, 0)
    misread = selective.counts(native, np.full(40, 4), mask)
    assert (misread["recordings_wrong"], misread["recordings_wrong_uncertain"], misread["recordings_right"]) == (1, 1, 0)  # a catch
    assert selective.counts(native, np.ones(40, int), np.zeros(40, bool))["recordings_right_uncertain"] == 0
    # a recording the decoder cuts is read by the majority of its samples, as in the scorecard: right at the middle is not enough
    cut = selective.counts(sequence("native", ("a", 2, 40)), np.array([3] * 14 + [2] * 12 + [3] * 14), mask)
    assert (cut["recordings_right"], cut["recordings_wrong"], cut["recordings_wrong_uncertain"], cut["piece_right_kept"]) == (0, 1, 1, 1)
    spliced = sequence("multi", ("a", 1, 20), ("b", 1, 20), ("c", 4, 20))  # a join of one view at 20, a view change at 40
    mask = np.zeros(60, bool)
    mask[24] = True  # 0.4 s after the join
    decided = np.array([1] * 40 + [4] * 20)
    c = selective.counts(spliced, decided, mask)
    assert (c["same_joins"], c["same_joins_uncertain"], c["changes"], c["changes_located"]) == (1, 1, 1, 0)
    mask[24], mask[25] = False, True  # 0.5 s after: outside
    assert selective.counts(spliced, decided, mask)["same_joins_uncertain"] == 0
    mask[39] = True  # the last sample before the change
    assert selective.counts(spliced, decided, mask)["changes_located"] == 1


def test_labels_with_the_uncertain_mark_are_refused(stream):
    seq, decided, band = stream
    with pytest.raises(ValueError, match="mask"):
        selective.counts(seq, np.where(band, -1, decided), band)
    with pytest.raises(ValueError, match="60 samples"):
        selective.counts(seq, decided[:50], band[:50])


def test_the_metrics_for_committed_labels_refuse_the_uncertain_mark(stream):
    """An uncertain sample has no view; a table indexed by view would silently count it as the last one."""
    seq, decided, band = stream
    marked = np.where(band, -1, decided)
    with pytest.raises(ValueError, match="uncertain"):
        scorecard.stream_counts(seq.labels, seq.pieces, marked)
    native = sequence("native", ("n", 1, 30), name="n0")
    with pytest.raises(ValueError, match="uncertain"):
        metrics.evaluate({seq.id: marked, native.id: np.ones(30, int)}, (seq, native))
    assert scorecard.stream_counts(seq.labels, seq.pieces, decided)["wrong"] == 22  # committed labels pass as before


def test_the_low_confidence_reference_declines_the_least_confident_samples():
    wrong = np.array([True, False, True, False, False, False])
    confidence = np.array([0.3, 0.9, 0.8, 0.4, 0.95, 0.99])
    assert selective.error_at_equal_coverage(wrong, confidence, 0) == pytest.approx(2 / 6)
    assert selective.error_at_equal_coverage(wrong, confidence, 2) == pytest.approx(1 / 4)  # 0.3 and 0.4 go: one wrong, one right
    assert np.isnan(selective.error_at_equal_coverage(wrong, confidence, 6))


def test_the_report_pools_by_kind_and_gives_intervals():
    rng = np.random.default_rng(0)
    sequences, decided, masks, bands, confidence = [], {}, {}, {}, {}
    for k in range(12):
        seq = sequence("multi", (f"a{k}", 0, 20), (f"b{k}", 2, 20), (f"c{k}", 3, 20), name=f"m{k}")
        labels = seq.labels.copy()
        if k % 3 == 0:
            labels[40:] = 2
        mask = np.zeros(60, bool)
        mask[18 + int(rng.integers(0, 3)):24] = True
        sequences.append(seq)
        decided[seq.id], masks[seq.id], bands[seq.id] = labels, mask, mask
        confidence[seq.id] = np.where(labels != seq.labels, 0.5, 0.9)  # a confidence that knows the wrong samples
    native = sequence("native", ("n", 1, 30), name="n0")
    sequences.append(native)
    decided[native.id], masks[native.id], bands[native.id], confidence[native.id] = np.ones(30, int), np.zeros(30, bool), np.zeros(30, bool), np.ones(30)
    report = selective.report(tuple(sequences), decided, masks, bands, confidence, draws=200)
    assert set(report) == {"native", "multi", "all"}  # kinds without a sequence are left out
    multi = report["multi"]
    assert multi["sequences"] == 12 and multi["counts"]["n"] == 720 and multi["counts"]["piece_wrong_kept"] == 4
    low, high = multi["intervals"]["error_labelled"]
    assert low <= multi["ratios"]["error_labelled"] <= high and low < high
    # every number about samples has an interval; the two about pieces get theirs over recordings, in piece_summary
    assert set(multi["intervals"]) == (set(multi["ratios"]) - set(selective.PIECE_NUMBERS)) | {"error_low_confidence", "error_labelled_minus_low_confidence"}
    assert len(multi["intervals"]["purity"]) == 5 and multi["intervals"]["purity"][1] is None  # no sample is given view 1
    low, high = multi["intervals"]["purity"][2]
    assert low <= multi["ratios"]["purity"][2] <= high and low < high
    assert multi["error_low_confidence"] < multi["ratios"]["error_labelled"]  # it declines only wrong samples here
    assert multi["error_labelled_minus_low_confidence"] == pytest.approx(multi["ratios"]["error_labelled"] - multi["error_low_confidence"])
    low, high = multi["intervals"]["error_labelled_minus_low_confidence"]
    assert 0 < low <= multi["error_labelled_minus_low_confidence"] <= high  # in every draw the rule is worse than a confidence that knows
    low, high = multi["intervals"]["error_low_confidence"]
    assert low <= multi["error_low_confidence"] <= high
    assert report["all"]["counts"]["n"] == 750 and report["native"]["ratios"]["uncertain_share"] == 0.0
    assert report["native"]["intervals"]["error_if_forced"] is None  # nothing was declined there: 0 of 0 has no interval
    assert report["native"]["intervals"]["error_low_confidence"] is None and report["native"]["error_low_confidence"] == 0.0
    assert multi["interval_lengths"]["count"] == 12 and multi["interval_lengths"]["median_s"] > 0
    again = selective.report(tuple(sequences), decided, masks, bands, confidence, draws=200)
    assert again["multi"]["intervals"] == multi["intervals"]  # the same draws every time


def test_pieces_are_counted_by_recording_and_the_score_is_drawn_over_recordings():
    rows = []
    cells = (("r1", True, True, 6), ("r2", True, True, 1), ("r3", True, False, 2), ("r4", False, True, 5), ("r5", False, False, 30))
    for name, wrong, declined, times in cells:
        rows += [{"recording": name, "kind": "multi", "wrong": wrong, "declined": declined, "beyond_band": True, "empty": False}] * times
    rows.append({"recording": "r6", "kind": "multi", "wrong": True, "declined": True, "beyond_band": False, "empty": True})  # the band's own
    summary = selective.piece_summary(rows, draws=500)
    assert (summary["pieces"], summary["recordings"], summary["empty_pieces"], summary["empty_recordings"]) == (45, 6, 1, 1)
    assert summary["accuracy_published"] == pytest.approx(35 / 45) and summary["accuracy_labelled"] == pytest.approx(30 / 32)
    for name in ("published", "labelled"):  # drawn over the six recordings
        low, high = summary[f"accuracy_{name}_interval"]
        assert low <= summary[f"accuracy_{name}"] <= high and low < high
    assert summary["all"]["wrong_declined"] == {"pieces": 8, "recordings": 3} and summary["beyond_band"]["wrong_declined"] == {"pieces": 7, "recordings": 2}
    assert summary["beyond_band"]["right_declined"] == {"pieces": 5, "recordings": 1} and summary["beyond_band"]["wrong_kept"] == {"pieces": 2, "recordings": 1}
    assert summary["score"] == pytest.approx(7 - 5 / 5)  # the piece the band declined does not count
    low, high = summary["score_p5_p95"]
    assert low < summary["score"] < high and low <= 0  # six of the seven catches are one recording: without it nothing is left
    nothing = selective.piece_summary([], draws=10)
    assert nothing["score"] == 0.0 and nothing["accuracy_published"] is None and nothing["accuracy_published_interval"] is None


def test_a_number_that_rests_on_five_or_fewer_has_no_interval():
    """Twenty streams with two wrong samples between them, both declined: 2 of 2 is a count, not a share with an interval."""
    sequences, decided, masks = [], {}, {}
    for k in range(20):
        seq = sequence("multi", (f"a{k}", 0, 30), (f"b{k}", 2, 30), name=f"m{k}")
        labels, mask = seq.labels.copy(), np.zeros(60, bool)
        if k < 2:
            labels[30], mask[30] = 0, True  # the cut is one sample late, and that sample is declined
        sequences.append(seq)
        decided[seq.id], masks[seq.id] = labels, mask
    multi = selective.report(tuple(sequences), decided, masks, draws=100)["multi"]
    assert (multi["counts"]["wrong"], multi["counts"]["wrong_uncertain"]) == (2, 2) and multi["ratios"]["errors_declined"] == 1.0
    assert multi["intervals"]["errors_declined"] is None and multi["intervals"]["error_if_forced"] is None
    assert multi["intervals"]["error_labelled"] is None  # 0 wrong among the labelled
    low, high = multi["intervals"]["coverage"]
    assert low <= multi["ratios"]["coverage"] <= high


def test_the_low_confidence_reference_in_a_draw_is_the_reference_of_the_drawn_sequences():
    rng = np.random.default_rng(3)
    sizes = (7, 4, 9, 5)
    owner = np.concatenate([np.full(size, k) for k, size in enumerate(sizes)])
    wrong, confidence = rng.random(owner.size) < 0.4, rng.random(owner.size).round(1)  # rounded: with ties
    weights = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 3, 0, 1], [0, 0, 0, 4], [1, 2, 1, 0]], float)
    declined = np.array([6, 9, 0, 30, 11], float)
    drawn = selective.error_at_equal_coverage_drawn(wrong, confidence, owner, weights, declined)
    assert drawn[0] == pytest.approx(selective.error_at_equal_coverage(wrong, confidence, 6))  # every sequence once: the plain reference
    for d in (1, 2, 4):  # the same number from the samples written out as often as their sequence is drawn
        copies = np.repeat(np.arange(owner.size), weights[d, owner].astype(int))
        order = copies[np.argsort(confidence[copies], kind="stable")][int(declined[d]):]
        assert drawn[d] == pytest.approx(wrong[order].mean())
    assert np.isnan(drawn[3])  # more declined than there are samples in the draw
