"""EV9V complementarity and fixed fusion (2026-10-05 plan, tasks 1-2): the plan's required checks on the small
synthetic stream of tests/test_windowed.py. A misaligned time axis stops the run; mapped B7 rows are finite and
normalised; fusion weights 1/0 and 0/1 give back each teacher; test inputs are refused; the calibration rule has no
fallback; fixed and newly wrong items add up to the totals."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.config import REPO_ROOT
from echo_routing.evaluate.complementarity import (
    AlignmentError,
    InputError,
    changes,
    check_alignment,
    check_distribution,
    check_no_test,
    check_sources,
    concentration,
    fuse,
    map_b7,
    mapping_matrix,
    native_refs,
    normalise_rows,
    recipe_refs,
    select_setting,
)
from echo_routing.features.cache import VideoFeatures
from echo_routing.features.extract import VARIANT_GAMMA
from echo_routing.temporal.windowed import RAW9_ORDER, RAW9_TO_FAMILY5, stream_plan

CFG = yaml.safe_load((REPO_ROOT / "configs/experiments/ev9v_complementarity_v1.yaml").read_text())
CLS = CFG["classes"]


def vf(video_id, n, shift=0):
    return VideoFeatures(video_id, np.arange(n) * 3 + shift, np.arange(n) / 10, np.zeros((n, 4)), np.full((n, 5), 0.2), np.zeros((n, 5)))


RECIPE = Recipe("x", "multi", (Fragment("a", 0, "orig", 2, 5), Fragment("b", 2, "gamma090", 0, 2)))


def plan_of(recipe):
    return [[r.video_id, r.frame_idx, r.gamma] for r in stream_plan(recipe, lambda v, var: vf(v, 10))]


def test_a_misaligned_time_axis_stops_the_run():
    plan = plan_of(RECIPE)
    check_alignment(recipe_refs(RECIPE, lambda v, var: vf(v, 10)), plan, VARIANT_GAMMA, "x")
    native = vf("c", 4)
    check_alignment(native_refs(native), [["c", int(f), 1.0] for f in native.frame_idx], VARIANT_GAMMA, "c")
    bad = {"one sample short": (recipe_refs(RECIPE, lambda v, var: vf(v, 10))[:-1], plan),
           "B7 shifted by one sample": (recipe_refs(RECIPE, lambda v, var: vf(v, 10)), [plan[0]] + plan[:-1]),
           "a variant cached on other frames": (recipe_refs(RECIPE, lambda v, var: vf(v, 10, shift=1 if var != "orig" else 0)), plan),
           "a sample read without its gamma": (recipe_refs(RECIPE, lambda v, var: vf(v, 10)), plan[:3] + [["b", 0, 1.0]] + plan[4:])}
    for name, (e0, b7) in bad.items():
        with pytest.raises(AlignmentError):
            check_alignment(e0, b7, VARIANT_GAMMA, name)
    with pytest.raises(AlignmentError):
        fuse(np.full((5, 5), 0.2), np.full((6, 5), 0.2), 0.5, 0.5)


def test_mapped_b7_rows_are_finite_normalised_and_follow_the_frozen_map():
    m = mapping_matrix(CLS["b7_raw9_order"], CLS["b7_raw9_to_family5"], CLS["family5"])
    frozen = np.zeros_like(m)
    for j, name in enumerate(RAW9_ORDER):
        if RAW9_TO_FAMILY5[name] is not None:
            frozen[j, RAW9_TO_FAMILY5[name]] = 1.0
    assert CLS["b7_raw9_order"] == list(RAW9_ORDER) and np.array_equal(m, frozen)
    raw = np.random.default_rng(0).dirichlet(np.ones(9), size=20)
    mapped, dropped = map_b7(raw, m, stored=(raw @ m).astype(np.float32))
    assert np.allclose(dropped, raw[:, RAW9_ORDER.index("PMPALA")])
    p = normalise_rows(mapped, CLS["b7_min_mapped_mass"])
    check_distribution(p, 1e-9)
    assert np.array_equal(p.argmax(1), mapped.argmax(1))
    nan = raw.copy(); nan[3, 2] = np.nan
    only_pmpala = np.zeros((2, 9)); only_pmpala[:, RAW9_ORDER.index("PMPALA")] = 1.0
    for fn in (lambda: map_b7(raw, m, stored=np.roll(raw @ m, 1, axis=1)), lambda: map_b7(nan, m),
               lambda: normalise_rows(map_b7(only_pmpala, m)[0], CLS["b7_min_mapped_mass"]),
               lambda: check_distribution(np.full((3, 5), 0.18), CLS["e0_sum_tolerance"])):
        with pytest.raises(InputError):
            fn()


def test_fusion_weights_one_and_zero_give_back_each_teacher():
    rng = np.random.default_rng(1); p, r = rng.dirichlet(np.ones(5), size=30), rng.dirichlet(np.ones(5), size=30)
    assert np.array_equal(fuse(p, r, 1.0, 0.0), p) and np.array_equal(fuse(p, r, 0.0, 1.0), r)
    w = CFG["fusion"]["weights"]
    q = fuse(p, r, w["E0"], w["B7"])
    check_distribution(q, 1e-9)
    assert np.allclose(q, (p + r) / 2)
    for bad in ((0.6, 0.6), (1.2, -0.2)):
        with pytest.raises(ValueError):
            fuse(p, r, *bad)


def test_test_clips_and_forbidden_sources_are_refused():
    check_no_test({"v_1", "v_2"}, {"t_1"})
    with pytest.raises(InputError):
        check_no_test({"v_1", "t_1"}, {"t_1"})
    forbidden = CFG["forbidden_input_substrings"]
    check_sources([CFG["inputs"]["protocol"]["path"], CFG["inputs"]["B7"]["runner_config"], CFG["inputs"]["B7"]["pred_dir"]], forbidden)
    for bad in ("data/manifests/pediatric/pediatric-external-v1/recipes_test_exam.json", "cache_v2/features/ped_adapt_raw/x.npz",
                "data/window_plans/old/multi_test.json"):
        with pytest.raises(InputError):
            check_sources([bad], forbidden)


def test_calibration_rule_has_no_fallback_and_breaks_ties_by_order():
    def rec(order, f1, recall, seg):
        return {"order": order, "native_macro_f1": f1, "boundary_recall_multi": recall, "segment_f1_50_multi": seg}
    rule = CFG["fusion"]["rule"]
    rows = [rec(0, 0.99, 0.94, 0.99), rec(1, 0.97, 0.99, 0.98), rec(2, 0.985, 0.96, 0.95), rec(3, 0.981, 0.97, 0.95), rec(4, 0.99, None, 0.999)]
    sel = select_setting(rows, rule["native_macro_f1_min"], rule["boundary_recall_min"])
    assert sel["status"] == "feasible" and sel["chosen"]["order"] == 2 and sel["n_feasible"] == 2
    none = select_setting(rows[:2], rule["native_macro_f1_min"], rule["boundary_recall_min"])
    assert none["status"] == "infeasible" and none["chosen"] is None


def test_fixed_and_new_items_add_up_to_the_totals():
    before = {"u1": False, "u2": True, "u3": False, "u4": True}; after = {"u1": True, "u2": False, "u3": True, "u4": True}
    fixed, new = changes(before, after)
    assert fixed == ["u1", "u3"] and new == ["u2"] and sum(after.values()) - sum(before.values()) == len(fixed) - len(new)
    c = concentration([["a", "b"], ["a"]], [["c"]])
    assert c["net"] == 1 and c["top1"] == {"recordings": ["a"], "fixed_involving": 2, "net_without": -1}
