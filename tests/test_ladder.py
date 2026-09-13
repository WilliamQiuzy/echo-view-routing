from __future__ import annotations

import numpy as np
import pytest

from echo_routing.compose.recipe_builder import Pool, make_pair_recipes
from echo_routing.compose.stream_assembler import render
from echo_routing.config import load_config
from echo_routing.evaluate.ladder import NativeCine, evaluate_b_file, evaluate_method
from echo_routing.evaluate.report import bfile_rows, ladder_rows, write_ladder
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.baselines.registry import decode

C = 5


def synth(video_id, label, n, rng, variant="orig"):
    conf = 0.85 if variant == "orig" else 0.75
    prob = rng.dirichlet(np.ones(C) * 0.3, size=n) * (1 - conf)
    prob[:, label] += conf
    prob /= prob.sum(1, keepdims=True)
    return VideoFeatures(video_id, np.arange(n) * 3, np.arange(n) / 10, np.zeros((n, 8)), prob, np.log(prob))


@pytest.fixture(scope="module")
def world():
    cfg = load_config("configs/base.yaml", overrides={"recipes": {"n_per_cell": 3, "min_len": 10, "max_len": 20}})
    rng = np.random.default_rng(0)
    ids = [f"v{i}" for i in range(60)]; labels = [i % C for i in range(60)]; n = [40] * 60
    feats = {(v, var): synth(v, l, 40, rng, var) for v, l in zip(ids, labels) for var in ("orig", "gamma090")}
    loader = lambda v, var: feats[(v, var)]
    natives = [NativeCine(v, l, feats[(v, "orig")].prob, feats[(v, "orig")].feat) for v, l in zip(ids, labels)]
    pool = Pool.from_index(ids, labels, n)
    val = [render(r, loader) for r in make_pair_recipes(pool, 3, np.random.default_rng(1), min_len=10, max_len=20)]
    test = [render(r, loader) for r in make_pair_recipes(pool, 3, np.random.default_rng(2), min_len=10, max_len=20)]
    return cfg, natives, val, test


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "b3"])
def test_registry_returns_label_per_sample(world, method):
    cfg, natives, _, _ = world
    labels = decode(method, natives[0].prob, cfg)
    assert labels.shape == (natives[0].prob.shape[0],)


def test_unknown_method(world):
    cfg, natives, _, _ = world
    with pytest.raises(KeyError):
        decode("nope", natives[0].prob, cfg)


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "b3"])
def test_evaluate_method_schema(world, method):
    cfg, natives, val, test = world
    m = evaluate_method(method, natives[:30], natives[30:], val, test, C, cfg)
    assert m["method_id"] == method
    assert set(m["constructed"]["cells"]) <= {"same_none", "same_edit", "diff_none", "diff_edit"}
    assert "0.5" in m["constructed"]["boundary"]
    pol = m["policy"]["by_target"]["0.05"]
    assert pol["status"] in ("ok", "target_unmet")
    if pol["status"] == "ok":
        assert m["constructed"]["routing"]["coverage"] > 0
    assert m["native"]["test"]["cine"]["accuracy"] > 0.9  # synthetic data is easy


def test_b_file_and_report(world, tmp_path):
    cfg, natives, val, test = world
    bf = evaluate_b_file(natives[:30], natives[30:], C, cfg)
    assert bf["method_id"] == "b_file" and "0.05" in bf["policy"]["by_target"]
    b1 = evaluate_method("b1", natives[:30], natives[30:], val, test, C, cfg)
    rows = ladder_rows([bf, b1]); assert len(rows) == 1 and rows[0]["method"] == "b1"
    assert len(bfile_rows([bf, b1])) == 3
    out = write_ladder([bf, b1], tmp_path)
    assert out.is_file() and "b1" in out.read_text() and (tmp_path / "b_file.csv").is_file()


def test_b4_decoder_via_registry(world, tmp_path):
    torch = pytest.importorskip("torch")
    from echo_routing.temporal.baselines.b4_mstcn import train_mstcn
    from echo_routing.temporal.baselines.registry import decode_with_prob
    cfg, natives, val, test = world
    cfg4 = dict(cfg, mstcn={"stages": 1, "layers": 2, "channels": 8, "lr": 0.01, "epochs": 2, "batch_size": 4, "checkpoint": None})
    best = train_mstcn(val, test, C, cfg4, tmp_path / "b4", torch.device("cpu"))
    cfg4["mstcn"]["checkpoint"] = str(best)
    labels, p = decode_with_prob("b4", test[0].prob, cfg4, test[0].feat)
    assert labels.shape == (test[0].prob.shape[0],) and p.shape == test[0].prob.shape
    with pytest.raises(KeyError):
        decode_with_prob("b4", test[0].prob, cfg, test[0].feat)
    m = evaluate_method("b4", natives[:5], natives[5:10], val[:4], test[:4], C, cfg4)
    assert m["method_id"] == "b4"
