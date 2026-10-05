"""B7 window-target check (configs/experiments/b7_window_target_v1.yaml): the windows are the runner's own, the
categories follow recordings and labels, the probe keeps the target's centre frames, and the ideal repair touches only
the chosen windows' samples."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from echo_routing.config import REPO_ROOT
from echo_routing.evaluate.failure_units import session_of
from echo_routing.evaluate.window_target import (CATEGORIES, category, middle, nearest_centre, pick_context, probe_window,
                                                 repair, share_bin, window_frames)
from echo_routing.temporal.windowed import window_centres
from scripts.run_windowed_echoviewclip import window_samples as runner_samples

CFG = yaml.safe_load((REPO_ROOT / "configs/experiments/b7_window_target_v1.yaml").read_text())


def test_windows_are_the_runners_frames():
    for n in (9, 15, 16, 37, 120):
        for c in window_centres(n, 5):
            samp = runner_samples(int(c), n, 16)
            assert window_frames(int(c), n, 16, 16).tolist() == samp[np.linspace(0, len(samp) - 1, 16).round().astype(int)].tolist()
    assert window_frames(20, 40, 16, 16).tolist() == list(range(12, 28))


def test_categories_follow_recordings_and_labels():
    assert category(["a_1"] * 16, [2] * 16) == CATEGORIES[0]
    assert category(["a_1"] * 8 + ["b_1"] * 8, [2] * 16) == CATEGORIES[1]
    assert category(["a_1"] * 8 + ["b_1"] * 8, [2] * 8 + [3] * 8) == CATEGORIES[2]
    assert [share_bin(k) for k in (3, 7, 8, 9, 11, 12, 15)] == ["<= 7 of 16"] * 2 + ["8 of 16"] + ["9-11 of 16"] * 2 + ["12-15 of 16"] * 2


def test_probe_keeps_the_targets_centre_frames_and_carries_references_only():
    target = middle([["t_1", 3 * i, 1.0] for i in range(40)])
    context = middle([["c_1", 3 * i, 1.0] for i in range(21)])
    assert target[0] == ["t_1", 36, 1.0] and context[0] == ["c_1", 6, 1.0]
    for m in CFG["probe"]["centre_frames"]:
        w = probe_window(target, context, m)
        lo, hi = 8 - m // 2, 8 + m // 2
        assert w[lo:hi] == target[lo:hi] and w[:lo] + w[hi:] == context[:lo] + context[hi:]
        assert all(isinstance(r, list) and len(r) == 3 for r in w)
    for bad in (0, 3, 18):
        with pytest.raises(ValueError):
            probe_window(target, context, bad)
    with pytest.raises(ValueError):
        middle([["s_1", 0, 1.0]] * 15)


def test_context_choice_is_seeded_and_avoids_the_targets_session():
    cands = ["2022-01-01_x_1", "2022-01-01_x_2", "2022-02-02_y_1", "2022-03-03_z_4"]
    picks = {pick_context(cands, {"2022-01-01_x"}, np.random.default_rng([1, 0, 3, 2]), session_of) for _ in range(5)}
    assert len(picks) == 1 and session_of(picks.pop()) != "2022-01-01_x"
    assert pick_context(cands[:2], {"2022-01-01_x"}, np.random.default_rng(0), session_of) is None


def test_repair_changes_only_the_chosen_windows_samples():
    n = 23; centres = window_centres(n, 5); labels = np.array([0] * 11 + [3] * 12)
    prob = np.random.default_rng(2).dirichlet(np.ones(5), size=n)
    out = repair(prob, centres, [2], labels)
    rows = nearest_centre(n, centres) == 2
    assert np.array_equal(out[~rows], prob[~rows])
    assert (out[rows].argmax(1) == labels[centres[2]]).all() and np.allclose(out[rows].sum(1), 1.0) and out[rows].max() == 1.0
    assert np.array_equal(repair(prob, centres, [], labels), prob)
