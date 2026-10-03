"""Round-4 failure units: reference units, the segment F1@50 matching, cause rules, spurious segments and repairs."""
from __future__ import annotations

import numpy as np

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.evaluate.failure_units import (match_changes, repair, segment_counts, session_of, stream_units,
                                                 unit_outcomes)
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import labels_to_segments


def _recipe(spec):
    """spec: [(label, length, video_id)] -> Recipe with consecutive source samples 0..length."""
    return Recipe("s", "multi", tuple(Fragment(v, lab, "orig", 0, n) for lab, n, v in spec))


def _units(labels_lengths):
    return stream_units(_recipe([(lab, n, f"sess_{i}") for i, (lab, n) in enumerate(labels_lengths)]))


def _ref(units):
    return np.concatenate([np.full(u.length, u.label) for u in units])


def test_units_merge_same_class_fragments_and_keep_sources_and_neighbours():
    units = stream_units(_recipe([(0, 20, "a_1"), (2, 10, "b_1"), (2, 15, "b_2"), (3, 12, "c_1")]))
    assert [(u.start, u.end, u.label) for u in units] == [(0, 20, 0), (20, 45, 2), (45, 57, 3)]
    assert [s[0] for s in units[1].sources] == ["b_1", "b_2"]
    assert (units[0].left, units[0].right, units[1].left, units[1].right, units[2].right) == (None, 2, 0, 3, None)
    assert session_of("2022-03-28_15-40-18_320x240_19") == "2022-03-28_15-40-18_320x240"


def test_matching_reproduces_segment_f1_counts():
    rng = np.random.default_rng(0)
    for _ in range(200):
        units = _units([(int(c), int(n)) for c, n in zip(rng.permutation(5)[:4], rng.integers(10, 40, 4))])
        ref = _ref(units); pred = ref.copy()
        for _ in range(int(rng.integers(0, 4))):                 # random corruptions: flips and shifted runs
            a = int(rng.integers(0, ref.size - 3)); pred[a:a + int(rng.integers(1, 15))] = int(rng.integers(0, 5))
        rows, _ = unit_outcomes(pred, units)
        m, n_pred, n_true = segment_counts(pred, units)
        assert sum(r["matched"] for r in rows) == m == segment_f1(labels_to_segments(pred), labels_to_segments(ref), 0.5)["matched"]
        assert n_pred == len(labels_to_segments(pred)) and n_true == len(units)


def test_short_unit_absorbed_by_its_neighbour_is_a_missed_cut():
    units = _units([(3, 40), (2, 10), (3, 40)]); pred = np.full(90, 3)
    rows, _ = unit_outcomes(pred, units)
    assert rows[1]["cause"] == "missed_cut" and rows[1]["majority"] == 3 and rows[1]["frac_label"] == 0.0


def test_absorbed_on_one_side_with_the_other_edge_found_is_still_a_missed_cut():
    # the change at the unit's right edge must not be counted for its left edge (one-to-one change matching)
    units = _units([(3, 40), (2, 10), (0, 40)]); pred = np.r_[np.full(50, 3), np.full(40, 0)]
    rows, _ = unit_outcomes(pred, units)
    assert rows[1]["cause"] == "missed_cut" and rows[1]["edge_dist"] == [None, 0]


def test_right_interval_with_wrong_class_is_a_label_error():
    units = _units([(0, 30), (2, 20), (1, 30)]); pred = _ref(units); pred[30:50] = 3
    rows, _ = unit_outcomes(pred, units)
    assert rows[1]["cause"] == "label" and rows[1]["best_label"] == 3 and rows[1]["best_iou"] == 1.0


def test_piece_inside_a_unit_is_an_extra_cut_and_a_hosted_spurious_segment():
    units = _units([(0, 30), (2, 30), (1, 30)]); pred = _ref(units); pred[38:52] = 4
    rows, spurious = unit_outcomes(pred, units)
    assert rows[1]["cause"] == "extra_cut" and rows[1]["extra_cuts"] == 2
    assert any(s["host"] == 1 and s["inside"] and s["label"] == 4 for s in spurious)


def test_edges_found_but_shifted_too_far_for_a_short_unit_is_a_boundary_shift():
    units = _units([(0, 30), (2, 10), (1, 30)]); pred = _ref(units); pred[30:37] = 0; pred[40:44] = 2
    rows, _ = unit_outcomes(pred, units)
    assert not rows[1]["matched"] and rows[1]["cause"] == "boundary_shift" and rows[1]["edge_dist"] == [7, 4]


def test_change_matching_is_one_to_one_and_greedy_by_distance():
    assert match_changes(np.array([12]), np.array([10, 13]), 5) == {13: 12}
    assert match_changes(np.array([4, 30]), np.array([10]), 5) == {}


def test_repairs():
    units = _units([(3, 40), (2, 10), (3, 40)]); ref = _ref(units); pred = np.full(90, 3)
    ideal = repair(pred, ref, units, [1], "ideal")
    assert segment_counts(ideal, units)[0] == 3
    scores = np.full((90, 5), 0.1); scores[40:50, 2] = 0.9            # the visual scores support the reference class
    seg = repair(pred, ref, units, [1], "segmentation", scores)
    assert seg[40:50].tolist() == [2] * 10 and segment_counts(seg, units)[0] == 3
    cls = repair(pred, ref, units, [1], "class")                      # the merged interval keeps its majority class
    assert cls.tolist() == pred.tolist() and segment_counts(cls, units)[0] == 0
    units2 = _units([(0, 30), (2, 20), (1, 30)]); pred2 = _ref(units2); pred2[30:50] = 3
    assert segment_counts(repair(pred2, _ref(units2), units2, [1], "class"), units2)[0] == 3
