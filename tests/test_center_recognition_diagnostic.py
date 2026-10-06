"""Recognition regression localisation (configs/experiments/center_recognition_diagnostic_v1.yaml): the grid paths
read only the window target samples and fill back as the old evaluation, native clips keep the majority vote, the
paired counts have the right direction and count sessions once, and the eight cells add up."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from echo_routing.evaluate.dev_metrics import native_table
from echo_routing.evaluate.window_target import nearest_centre
from echo_routing.temporal.dev_data import Seq
from echo_routing.temporal.windowed import window_centres
from scripts.diagnose_center_recognition import compare, eight_cells, native_bootstrap, native_right, to_grid


def test_grid_reads_only_target_samples_and_fills_back_like_the_old_rule():
    rng = np.random.default_rng(0)
    for n in (1, 4, 15, 16, 22, 23, 60):
        prob = rng.random((n, 5)); centres = window_centres(n, 5)
        g = to_grid(prob)
        assert g.shape == prob.shape and np.array_equal(g, prob[centres][nearest_centre(n, centres)])
        other = prob.copy(); keep = np.zeros(n, bool); keep[centres] = True; other[~keep] = rng.random(((~keep).sum(), 5))
        assert np.array_equal(to_grid(other), g)                       # only the target samples are read
    prob = np.arange(22 * 5, dtype=float).reshape(22, 5)
    assert window_centres(22, 5).tolist() == [2, 7, 12, 17, 21]
    assert np.array_equal(to_grid(prob)[19], prob[17])                   # a tie goes to the earlier centre
    assert np.array_equal(to_grid(prob[:4]), np.repeat(prob[2:3], 4, 0))  # short sequence: one centre


def test_native_clips_keep_the_majority_vote_with_ties_to_the_lower_class():
    natives = [Seq("a_1", True, np.zeros((6, 1)), np.full(6, 3), np.zeros(0, int)), Seq("b_1", True, np.zeros((4, 1)), np.full(4, 2), np.zeros(0, int))]
    tab = native_table({"a_1": np.array([3, 3, 2, 2, 1, 0]), "b_1": np.array([2, 3, 3, 2])}, natives)
    assert tab["majority"].tolist() == [2, 2]                            # 2-2 tie between A4C and A5C goes to A4C (2)
    assert native_right(tab) == {"a_1": False, "b_1": True}


def test_paired_counts_have_the_right_direction_and_count_sessions_once():
    ids = ["2022-01-01_x_1", "2022-01-01_x_2", "2022-02-02_y_1", "2022-03-03_z_1"]
    before = pd.DataFrame({"video_id": ids, "label": [3, 3, 2, 0], "majority": [2, 2, 2, 0]})
    after = pd.DataFrame({"video_id": ids, "label": [3, 3, 2, 0], "majority": [3, 3, 3, 0]})
    c = compare(after, before, {"draws": 50, "seed": 1})
    assert (c["fixed"], c["new"], c["fixed_sessions"], c["new_sessions"]) == (2, 1, 1, 1)
    assert c["fixed_by_class"] == {"A5C": 2} and c["new_by_class"] == {"A4C": 1}
    same = native_bootstrap(before, before, 50, 1)
    assert same["d_macro_f1"] == 0 and same["ci_macro_f1"] == [0.0, 0.0] and same["sessions"] == 3
    cells = eight_cells(native_right(before), native_right(after), native_right(after))
    assert sum(cells.values()) == 4 and cells["F- J+ T+"] == 2 and cells["F+ J- T-"] == 1 and cells["F+ J+ T+"] == 1
    with pytest.raises(ValueError):
        eight_cells({"a": True}, {"b": True}, {"a": True})
