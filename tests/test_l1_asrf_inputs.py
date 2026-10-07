"""Input checks of the L1 + ASRF round (scripts/prepare_l1_asrf.py) on synthetic data: the export file contract, the
clip sets, score computation, the A/B stream alignment and the cache key."""
import numpy as np
import pandas as pd
import pytest
import torch

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.compose.stream_assembler import render
from echo_routing.features.cache import VideoFeatures
from scripts.prepare_l1_asrf import cache_key, check_clip_sets, check_export_file, compare_streams, scores, stream_signature


def _export(n: int = 6, dim: int = 512) -> dict:
    rng = np.random.default_rng(0)
    return {"frame_idx": np.arange(n) * 3, **{f"{v}_feat": rng.standard_normal((n, dim)).astype(np.float16) for v in ("orig", "gamma090")},
            **{f"{v}_prob": np.full((n, 5), 0.2, np.float32) for v in ("orig", "gamma090")}}


def test_export_file_contract():
    z = _export(); check_export_file(z, np.arange(6) * 3)
    with pytest.raises(ValueError, match="missing gamma090_feat"):
        check_export_file({k: v for k, v in z.items() if k != "gamma090_feat"}, np.arange(6) * 3)
    with pytest.raises(ValueError, match="frame indices"):
        check_export_file(z, np.array([0, 6, 3, 9, 12, 15]))          # reordered
    with pytest.raises(ValueError, match="frame indices"):
        check_export_file(z, np.arange(7) * 3)                          # a frame missing from the export
    with pytest.raises(ValueError, match="expected \\(6, 512\\) float16"):
        check_export_file({**z, "orig_feat": z["orig_feat"][:, :256]}, np.arange(6) * 3)
    with pytest.raises(ValueError, match="float16"):
        check_export_file({**z, "orig_feat": z["orig_feat"].astype(np.float32)}, np.arange(6) * 3)
    bad = z["orig_feat"].copy(); bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        check_export_file({**z, "orig_feat": bad}, np.arange(6) * 3)


def test_clip_sets_refuse_missing_extra_and_test_rows():
    proto = {"train": {"clips": ["t1", "t2"]}, "validation": {"selection": {"clips": ["s1"]}, "calibration": {"clips": ["c1"]}}}
    idx = pd.DataFrame({"video_id": ["t1", "t2", "s1", "c1"], "split": ["train", "train", "validation", "validation"]})
    check_clip_sets(idx, proto)
    with pytest.raises(ValueError, match="missing"):
        check_clip_sets(idx.iloc[:3], proto)
    with pytest.raises(ValueError, match="no test row"):
        check_clip_sets(pd.concat([idx, pd.DataFrame({"video_id": ["x9"], "split": ["test"]})]), proto)
    with pytest.raises(ValueError, match="no test row"):
        check_clip_sets(pd.concat([idx, pd.DataFrame({"video_id": ["t9"], "split": ["train"]})]), proto)


def test_scores_are_the_head_on_the_stored_fp16_feature():
    torch.manual_seed(0); head = torch.nn.Linear(512, 5)
    f16 = np.random.default_rng(1).standard_normal((4, 512)).astype(np.float16)
    lg, pr = scores(head, f16, torch.device("cpu"))
    ref = head(torch.from_numpy(f16.astype(np.float32))).detach().numpy()
    assert np.allclose(lg, ref, atol=1e-6) and np.allclose(pr.sum(1), 1.0, atol=1e-6) and np.array_equal(pr.argmax(1), ref.argmax(1))


def _vf(vid: str, n: int, seed: int) -> VideoFeatures:
    rng = np.random.default_rng(seed)
    return VideoFeatures(vid, np.arange(n) * 3, np.arange(n) * 0.1, rng.standard_normal((n, 512)).astype(np.float32),
                         np.full((n, 5), 0.2, np.float32), np.zeros((n, 5), np.float32))


def _streams(feature_seed: int, labels=(2, 3)) -> list:
    stores = {("a", "orig"): _vf("a", 12, feature_seed), ("b", "gamma090"): _vf("b", 10, feature_seed + 1)}
    r = Recipe("r1", "cell", (Fragment("a", labels[0], "orig", 0, 8), Fragment("b", labels[1], "gamma090", 2, 9)))
    return [render(r, lambda v, var: stores[(v, var)])]


def test_streams_must_match_except_features():
    a, b = _streams(0), _streams(5)
    assert compare_streams(a, b) == 1                 # only the features differ
    assert stream_signature(a[0])["semantic"] == [8]
    with pytest.raises(ValueError, match="differs"):
        compare_streams(a, _streams(5, labels=(2, 2)))  # a different label is refused
    with pytest.raises(ValueError, match="streams"):
        compare_streams(a, a + a)


def test_cache_key_is_deterministic_and_model_specific():
    assert cache_key("abc", 0) == cache_key("abc", 0) and len(cache_key("abc", 0)) == 12
    assert cache_key("abc", 0) != cache_key("abd", 0) and cache_key("abc", 0) != cache_key("abc", 1)


def test_new_index_keeps_the_e1_row_order():
    from scripts.prepare_l1_asrf import index_order
    e1 = pd.DataFrame({"video_id": ["c", "a", "b"]})           # not alphabetical: the training bank follows this order
    rows = {"a": {}, "b": {}, "c": {}}
    assert index_order(e1, rows) == ["c", "a", "b"]
    with pytest.raises(ValueError, match="same set"):
        index_order(e1, {"a": {}, "b": {}})
    with pytest.raises(ValueError, match="same set"):
        index_order(e1, {"a": {}, "b": {}, "c": {}, "d": {}})
