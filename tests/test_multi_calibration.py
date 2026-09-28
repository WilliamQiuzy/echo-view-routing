"""Thresholds selected on multi-fragment validation streams, per-fragment count tables, and the paired bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from echo_routing.compose.recipe_builder import Pool, make_multi_recipes, make_pair_recipes
from echo_routing.compose.stream_assembler import render
from echo_routing.config import load_config
from echo_routing.evaluate.bootstrap import BASE_COLS, metrics, resample_weights, route_keys, summarise, unit_sums
from echo_routing.evaluate.ladder import NativeCine, evaluate_method
from echo_routing.features.cache import VideoFeatures

C = 5
TARGETS = {"0.05", "0.01", "0.1"}


def synth(video_id, label, n, rng, variant="orig"):
    conf = 0.6 if variant == "orig" else 0.5          # noisy enough that some thresholds bind
    prob = rng.dirichlet(np.ones(C) * 0.3, size=n) * (1 - conf)
    prob[:, label] += conf
    prob /= prob.sum(1, keepdims=True)
    return VideoFeatures(video_id, np.arange(n) * 3, np.arange(n) / 10, np.zeros((n, 8)), prob, np.log(prob))


@pytest.fixture(scope="module")
def world():
    cfg = load_config("configs/base.yaml", overrides={"recipes": {"n_per_cell": 3, "min_len": 10, "max_len": 20}})
    rng = np.random.default_rng(0)
    ids = [f"v{i}" for i in range(60)]; labels = [i % C for i in range(60)]
    feats = {(v, var): synth(v, l, 40, rng, var) for v, l in zip(ids, labels) for var in ("orig", "gamma090")}
    loader = lambda v, var: feats[(v, var)]  # noqa: E731
    natives = [NativeCine(v, l, feats[(v, "orig")].prob, feats[(v, "orig")].feat) for v, l in zip(ids, labels)]
    pool = Pool.from_index(ids, labels, [40] * 60)
    pairs = {s: [render(r, loader) for r in make_pair_recipes(pool, 3, np.random.default_rng(seed), min_len=10, max_len=20)]
             for s, seed in (("validation", 1), ("test", 2))}
    multi = {s: [render(r, loader) for r in make_multi_recipes(pool, 30, np.random.default_rng(seed), n_fragments=(3, 5),
                                                                min_len=10, max_len=20)]
             for s, seed in (("validation", 3), ("test", 4))}
    m = evaluate_method("b1", natives[:30], natives[30:], pairs["validation"], pairs["test"], C, cfg,
                        test_multi=multi["test"], val_multi=multi["validation"])
    return cfg, natives, pairs, multi, m


def test_policy_selected_on_multi_validation(world):
    *_, m = world
    assert m["policy_multi"]["selected_on"] == "multi_validation"
    assert set(m["policy_multi"]["by_target"]) == TARGETS
    keys = set(m["constructed_multi"]["routing_by"])
    assert {f"{p}@{t}" for t in TARGETS for p in ("pair", "multi", "p2")} == keys
    assert set(m["constructed"]["routing_by"]) == keys


def test_pair_tau_on_multi_matches_existing_field(world):
    *_, m = world
    old, new = m["constructed_multi"]["routing"], m["constructed_multi"]["routing_by"]["pair@0.05"]
    if old is None:
        assert new["tau"] is None
    else:
        assert new["coverage"] == pytest.approx(old["coverage"]) and new["tau"] == old["tau"]


@pytest.mark.parametrize("bank,block", [("pairs_test", "constructed"), ("multi_test", "constructed_multi")])
def test_fragment_rows_reproduce_pooled_metrics(world, bank, block):
    *_, m = world
    df = pd.DataFrame(m["_fragments"][bank])
    b = m[block]["boundary"]["0.5"]
    assert df["t_true"].sum() == b["n_true"] and df["p_pred"].sum() == b["n_pred"] and df["t_matched"].sum() == b["matched"]
    cells = m[block]["cells"]
    assert df["false_split"].sum() == sum(c["false_splits"] for c in cells.values())
    assert df["missed"].sum() == sum(c["missed"] for c in cells.values())
    assert df["same_join"].sum() == sum(c["same_view_joins"] for c in cells.values())
    point = metrics(df[BASE_COLS + [f"{p}|{k}" for k in route_keys(df) for p in ("acc", "wrong")]].sum().to_numpy(float),
                    BASE_COLS + [f"{p}|{k}" for k in route_keys(df) for p in ("acc", "wrong")])
    assert point["boundary_f1"] == pytest.approx(b["f1"])
    for k, r in m[block]["routing_by"].items():   # global ("tau") and per-class ("tau_by_class") entries alike
        if "coverage" in r:
            assert point[f"coverage|{k}"] == pytest.approx(r["coverage"])
            if r["risk"] is not None:
                assert point[f"risk|{k}"] == pytest.approx(r["risk"])


def test_without_multi_validation_is_backward_compatible(world):
    cfg, natives, pairs, _, _ = world
    m = evaluate_method("b1", natives[:30], natives[30:], pairs["validation"], pairs["test"], C, cfg)
    assert "policy_multi" not in m and "policy2" not in m and m["constructed_multi"] is None
    assert set(m["_fragments"]) == {"pairs_test"}
    assert set(m["constructed"]["routing_by"]) == {f"pair@{t}" for t in TARGETS}


def test_cine_units_are_fewer_than_fragments_when_reused(world):
    *_, m = world
    df = pd.DataFrame(m["_fragments"]["multi_test"])
    ids, X = unit_sums(df, "video_id", BASE_COLS)
    assert len(ids) < len(df) and X.shape == (len(ids), len(BASE_COLS))
    assert X.sum(0) == pytest.approx(df[BASE_COLS].sum().to_numpy(float))


def test_resample_weights_are_multiplicities():
    w = resample_weights(7, 50, np.random.default_rng(0))
    assert w.shape == (50, 7) and np.all(w.sum(1) == 7) and np.all(w >= 0)


def test_paired_bootstrap_self_difference_is_zero(world):
    *_, m = world
    df = pd.DataFrame(m["_fragments"]["multi_test"])
    cols = BASE_COLS + [f"{p}|{k}" for k in route_keys(df) for p in ("acc", "wrong")]
    _, X = unit_sums(df, "video_id", cols)
    w = resample_weights(X.shape[0], 200, np.random.default_rng(1))
    res = summarise({"a": X, "b": X.copy()}, cols, w, reference="a")
    e = res["b"]["boundary_f1"]
    assert e["diff"] == 0 and e["diff_ci95"] == [0, 0]
    assert e["ci95"][0] <= e["estimate"] <= e["ci95"][1]
    assert "diff" not in res["a"]["boundary_f1"]


def test_policy2_reported_and_reproducible_from_fragments(world):
    *_, m = world
    p2 = m["policy2"]
    assert p2["selected_on"] == "multi_validation" and p2["evidence"] == "b0" and p2["rule"] == "ucb"
    assert set(p2["by_target"]) == TARGETS
    for block, bank in (("constructed", "pairs_test"), ("constructed_multi", "multi_test")):
        df = pd.DataFrame(m["_fragments"][bank])
        for t in TARGETS:
            r = m[block]["routing_by"][f"p2@{t}"]
            assert set(r["tau_by_class"]) <= {str(c) for c in range(C)}
            assert df[f"acc|p2@{t}"].sum() / df["n"].sum() == pytest.approx(r["coverage"])


def test_frame_stage_and_fused_ids():
    from echo_routing.temporal.baselines.registry import evidence_source, parse_fused
    assert evidence_source("b4") == "b0" and evidence_source("vit_router") == "vit_frame" and evidence_source("b7") == "b7"
    assert evidence_source("vivit_fe@s1") == "vivit_fe@s1"
    assert parse_fused("b4+b7~0.25") == ("b4", "b7", 0.25) and parse_fused("b4+b7") == ("b4", "b7", 0.5)
    assert parse_fused("b4") is None and evidence_source("b4+b7~0.25") == "0.75*b0 + 0.25*b7"


def test_fused_decoder_keeps_boundaries_and_relabels_segments(world):
    from echo_routing.temporal.baselines.registry import decode_with_prob, evidence_prob
    from echo_routing.temporal.segments import labels_to_segments
    cfg, natives, *_ = world
    p = natives[3].prob
    labels, mix = decode_with_prob("b1+b2~1", p, cfg)
    for s in labels_to_segments(decode_with_prob("b1", p, cfg)[0]):
        assert np.all(labels[s.start:s.end] == p[s.start:s.end].mean(0).argmax())
    assert mix == pytest.approx(p) and evidence_prob("b1+b2~0.5", p, cfg) == pytest.approx(p)
