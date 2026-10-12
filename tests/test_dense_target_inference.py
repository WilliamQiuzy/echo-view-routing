"""Index, window and mapping checks of the dense target inference round (no GPU, no data).

Expected values are worked out by hand from the source rules (window_centres(n, 5), window_frames(c, n, 16, 16), the
middle target slot, nearest centre with ties to the earlier one), not copied from the implementation.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.dense_target_inference as dt  # noqa: E402


# ----------------------------------------------------------------------------------------------- centres
def test_old_centres_follow_the_source_rule():
    assert dt.old_centres(1).tolist() == [0]
    assert dt.old_centres(2).tolist() == [1]
    assert dt.old_centres(3).tolist() == [2]
    assert dt.old_centres(7).tolist() == [2, 6]
    assert dt.old_centres(10).tolist() == [2, 7]           # divisible by 5
    assert dt.old_centres(12).tolist() == [2, 7, 11]       # not divisible: the last sample is added
    assert dt.old_centres(15).tolist() == [2, 7, 12]
    assert dt.old_centres(16).tolist() == [2, 7, 12, 15]
    assert dt.old_centres(21).tolist() == [2, 7, 12, 17, 20]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 9, 10, 15, 16, 17, 20, 23, 64, 75])
def test_dense_centres_cover_every_sample_once_and_contain_the_old(n):
    old, new = dt.old_centres(n), dt.new_centres(n)
    assert set(old.tolist()) <= set(range(n))
    assert not set(old.tolist()) & set(new.tolist())
    assert sorted(old.tolist() + new.tolist()) == list(range(n))
    assert np.all(np.diff(new) > 0)


# ----------------------------------------------------------------------------------------------- windows and markers
def test_a_one_sample_sequence_repeats_its_frame_and_marks_the_middle():
    s, m = dt.window_spec(1, 0)
    assert s.tolist() == [0] * 16 and m == 8


def test_a_short_sequence_keeps_the_runner_repeats():
    s, m = dt.window_spec(5, 2)
    assert s.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    assert m == 8                                          # target slots 6-9, the middle one (index 2 of 4)
    assert dt.window_spec(5, 0)[1] == 1                    # target slots 0-1
    assert dt.window_spec(5, 4)[1] == 15                   # target slots 14-15


def test_windows_clamp_at_the_ends_of_a_long_sequence():
    s, m = dt.window_spec(40, 0)
    assert s.tolist() == list(range(16)) and m == 0
    s, m = dt.window_spec(40, 3)
    assert s.tolist() == list(range(16)) and m == 3
    s, m = dt.window_spec(40, 20)
    assert s.tolist() == list(range(12, 28)) and m == 8
    s, m = dt.window_spec(40, 39)
    assert s.tolist() == list(range(24, 40)) and m == 15


def _toy_sequences():
    # no "labels" key: building windows must not need labels
    return [{"id": "a", "kind": "native", "refs": [("v_1", i, 1.0) for i in range(7)]},
            {"id": "b", "kind": "multi", "refs": [("v_1", i, 1.0) for i in range(3)] + [("w_1", i, 0.7) for i in range(10)]}]


def test_dense_windows_read_references_only_and_flag_the_old_centres():
    seqs = _toy_sequences()
    refs = sorted({r for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    W = dt.dense_windows(seqs, at)
    assert W["seq"].tolist() == [0] * 7 + [1] * 13
    assert W["centre"].tolist() == list(range(7)) + list(range(13))
    assert W["ref"].shape == (20, 16)
    for i, (q, c) in enumerate(zip(W["seq"], W["centre"])):
        assert refs[W["ref"][i][W["marker"][i]]] == seqs[q]["refs"][c]      # the marked frame is the sample's own reference
    assert W["old"].tolist() == [c in (2, 6) for c in range(7)] + [c in (2, 7, 12) for c in range(13)]


def test_the_old_windows_are_exactly_the_dense_windows_at_old_centres():
    seqs = _toy_sequences()
    refs = sorted({r for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    W, O = dt.dense_windows(seqs, at), dt.sparse_windows(seqs, at)
    assert O["centre"].tolist() == [2, 6, 2, 7, 12] and O["seq"].tolist() == [0, 0, 1, 1, 1]
    for k in ("seq", "centre", "ref", "marker"):
        assert np.array_equal(O[k], W[k][W["old"]])
    # a stream window crossing the join carries each fragment's own gamma
    i = int(np.flatnonzero((W["seq"] == 1) & (W["centre"] == 2))[0])
    gammas = [refs[j][2] for j in W["ref"][i]]
    assert 1.0 in gammas and 0.7 in gammas


# ----------------------------------------------------------------------------------------------- mappings
def test_sparse_takes_the_nearest_centre_and_the_earlier_one_on_a_tie():
    centres = np.array([2, 7, 11]); w = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    p = dt.sparse_probs(12, centres, w)
    assert p.dtype == np.float32 and p.shape == (12, 5)
    assert [int(r.argmax()) for r in p] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]   # sample 9 ties 7 and 11 -> 7


def test_sparse_equals_the_source_per_sample_mapping():
    import scripts.report_raw_video_context as rr
    rng = np.random.default_rng(1); lg = (rng.normal(size=(5, 5)) * 4).astype(np.float32)
    W = {"seq": np.array([0, 0, 0, 1, 1]), "centre": np.array([2, 7, 11, 2, 6])}
    seqs = [{"id": "a", "labels": np.zeros(12, int)}, {"id": "b", "labels": np.zeros(7, int)}]
    src = rr.per_sample(W, lg, seqs); w = dt.softmax32(lg)
    assert np.array_equal(dt.sparse_probs(12, W["centre"][:3], w[:3]), src["a"])
    assert np.array_equal(dt.sparse_probs(7, W["centre"][3:], w[3:]), src["b"])


def test_softmax_is_the_source_float32_softmax():
    rng = np.random.default_rng(0); lg = (rng.normal(size=(7, 5)) * 5).astype(np.float32)
    p = dt.softmax32(lg)
    assert p.dtype == np.float32 and np.array_equal(p, torch.softmax(torch.from_numpy(lg), -1).numpy())


def test_interpolation_is_linear_between_centres_and_holds_the_ends():
    centres = np.array([2, 7]); w = np.array([[0.6, 0.4, 0, 0, 0], [0.1, 0.9, 0, 0, 0]], np.float32)
    p = dt.interp_probs(10, centres, w)
    assert p.dtype == np.float32 and p.shape == (10, 5)
    assert np.array_equal(p[2], w[0]) and np.array_equal(p[7], w[1])          # old centres unchanged, bit for bit
    assert np.array_equal(p[0], w[0]) and np.array_equal(p[1], w[0])          # first value held before the first centre
    assert np.array_equal(p[8], w[1]) and np.array_equal(p[9], w[1])          # last value held after the last centre
    want = w[0].astype(float) + (w[1].astype(float) - w[0].astype(float)) * 2 / 5
    np.testing.assert_allclose(p[4], want, rtol=0, atol=1e-7)
    np.testing.assert_allclose(p.sum(1), 1.0, rtol=0, atol=1e-6)


def test_interpolation_can_name_a_class_that_no_centre_names():
    # neither centre's argmax is class 2, but the midpoint's is: allowed and not assumed safe
    w = np.array([[0.45, 0.0, 0.40, 0.15, 0.0], [0.0, 0.45, 0.40, 0.0, 0.15]], np.float32)
    p = dt.interp_probs(13, np.array([2, 12]), w)
    assert int(p[2].argmax()) == 0 and int(p[12].argmax()) == 1 and int(p[7].argmax()) == 2


def test_interpolation_with_one_centre_is_constant():
    w = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], np.float32)
    assert np.array_equal(dt.interp_probs(3, np.array([2]), w), np.repeat(w, 3, 0))


def test_interpolation_refuses_rows_that_are_not_probabilities():
    with pytest.raises(ValueError):
        dt.interp_probs(10, np.array([2, 7]), np.array([[0.5, 0.5, 0, 0, 0], [0.5, 0.4, 0, 0, 0]], np.float32))   # sums to 0.9
    with pytest.raises(ValueError):
        dt.interp_probs(10, np.array([2, 7]), np.array([[1.2, -0.2, 0, 0, 0], [0.5, 0.5, 0, 0, 0]], np.float32))  # negative


# ----------------------------------------------------------------------------------------------- dense assembly
def test_dense_assembly_reuses_old_rows_and_gives_one_output_per_sample():
    n = 12; old, new = dt.old_centres(n), dt.new_centres(n)
    old_v = np.arange(len(old) * 5, dtype=np.float32).reshape(-1, 5) + 100
    new_v = -np.arange(len(new) * 5, dtype=np.float32).reshape(-1, 5)
    out, is_old = dt.assemble(n, old, old_v, new, new_v)
    assert out.shape == (n, 5) and out.dtype == np.float32
    assert np.array_equal(out[old], old_v) and np.array_equal(out[new], new_v)
    assert is_old.tolist() == [c in (2, 7, 11) for c in range(n)]
    with pytest.raises(ValueError):
        dt.assemble(n, old, old_v, np.r_[new, old[:1]], np.r_[new_v, old_v[:1]])   # an old centre inferred again
    with pytest.raises(ValueError):
        dt.assemble(n, old, old_v, new[1:], new_v[1:])                              # a sample without an output


# ----------------------------------------------------------------------------------------------- sources
def test_a_pinned_file_with_another_digest_is_refused(tmp_path):
    (tmp_path / "x.npz").write_bytes(b"abc")
    good = {"path": "x.npz", "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert dt.pinned(tmp_path, good) == tmp_path / "x.npz"
    with pytest.raises(SystemExit):
        dt.pinned(tmp_path, {**good, "sha256": "0" * 64})
    with pytest.raises(SystemExit):
        dt.pinned(tmp_path, {"path": "missing.npz", "sha256": good["sha256"]})
