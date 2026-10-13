"""Whole-sequence video features for the temporal heads (video_feature_temporal_v1): windows from assembled references,
label isolation, nearest-centre mapping, the float32 batch adapter and the training-only index read. Expected values are
worked out by hand from the source rules, not copied from the implementation."""
from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import echo_routing.temporal.video_sequence_features as vf  # noqa: E402
from echo_routing.compose.recipe_builder import Fragment  # noqa: E402
from echo_routing.temporal.dev_data import TrainData, collate  # noqa: E402


# ----------------------------------------------------------------------------------------------- references
def test_native_references_use_the_cached_frame_indices():
    assert vf.native_refs("v_1", np.array([0, 3, 6])) == [("v_1", 0, 1.0), ("v_1", 3, 1.0), ("v_1", 6, 1.0)]


def test_stream_references_follow_the_recipe_with_each_fragment_gamma():
    frags = (Fragment("a_1", 2, "orig", 1, 4), Fragment("b_1", 3, "gamma090", 0, 2))
    idx = {"a_1": np.array([0, 3, 6, 9, 12]), "b_1": np.array([5, 8, 11])}
    refs = vf.stream_refs(frags, idx.__getitem__)
    assert refs == [("a_1", 3, 1.0), ("a_1", 6, 1.0), ("a_1", 9, 1.0), ("b_1", 5, 0.9), ("b_1", 8, 0.9)]


def test_same_view_joins_are_joins_but_not_label_changes():
    frags = (Fragment("a_1", 2, "orig", 0, 10), Fragment("b_1", 2, "orig", 0, 10), Fragment("c_1", 3, "orig", 0, 10))
    labels, joins = vf.stream_labels_joins(frags)
    assert labels.tolist() == [2] * 20 + [3] * 10
    assert joins.tolist() == [10, 20]
    assert vf.label_changes(labels).tolist() == [20]           # the same-view join at 10 is no true boundary


# ----------------------------------------------------------------------------------------------- windows and mapping
def test_windows_cross_a_join_and_carry_both_gammas():
    refs = [("a_1", i, 1.0) for i in range(8)] + [("b_1", i, 0.9) for i in range(8)]
    w = vf.windows(len(refs))
    assert w["centres"].tolist() == [2, 7, 12, 15]
    s = w["samples"][w["centres"].tolist().index(7)]           # window at 7 of 16 samples: samples 0-15
    assert s.tolist() == list(range(16)) and w["markers"][1] == 7
    gam = {refs[i][2] for i in s}
    assert gam == {1.0, 0.9}


def test_short_sequences_repeat_samples_and_mark_the_middle_target_slot():
    w = vf.windows(5)
    assert w["centres"].tolist() == [2]                        # the last sample is only 2 beyond the centre: no end window
    assert w["samples"][0].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    assert w["markers"].tolist() == [8]
    seven = vf.windows(7)
    assert seven["centres"].tolist() == [2, 6] and seven["markers"].tolist() == [5, 15]   # target slots 4-6, then 14-15
    one = vf.windows(1)
    assert one["centres"].tolist() == [0] and one["samples"][0].tolist() == [0] * 16 and one["markers"].tolist() == [8]


def test_nearest_centre_mapping_ties_to_the_earlier_centre_and_keeps_float32():
    wfeat = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    out = vf.map_to_samples(12, np.array([2, 7, 11]), wfeat)
    assert out.dtype == np.float32 and out.shape == (12, 4)
    rows = [int(np.flatnonzero((wfeat == r).all(1))[0]) for r in out]
    assert rows == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]       # sample 9 ties 7 and 11 -> 7


def _fake_embed(windows_refs, markers):
    """A deterministic stand-in for model.embed: a function of the 16 references and the marker only."""
    out = []
    for refs, m in zip(windows_refs, markers):
        key = np.array([zlib.crc32(f"{r[0]}|{int(r[1])}|{float(r[2]):.6f}".encode()) % 997 for r in refs], np.float64)
        out.append(np.concatenate([key[:4], [m, key.sum() % 31]]).astype(np.float32))
    return np.stack(out)


def test_features_depend_on_the_references_only():
    refs = [("a_1", i, 1.0) for i in range(9)] + [("b_1", i, 0.9) for i in range(14)]
    f1 = vf.sequence_features(refs, _fake_embed)
    f2 = vf.sequence_features(list(refs), _fake_embed)          # the same references; labels, joins and flags are never passed
    assert np.array_equal(f1["samples"], f2["samples"]) and f1["samples"].shape == (23, 6)
    w = vf.windows(23)
    assert np.array_equal(f1["centres"], w["centres"]) and f1["window"].shape == (len(w["centres"]), 6)
    assert np.array_equal(f1["samples"], vf.map_to_samples(23, w["centres"], f1["window"]))
    other = [("a_1", i, 1.0) for i in range(9)] + [("b_1", i, 1.0) for i in range(14)]   # another gamma is another input
    assert not np.array_equal(vf.sequence_features(other, _fake_embed)["samples"], f1["samples"])


def test_the_marked_frame_is_the_requested_sample():
    refs = [("a_1", 10 * i, 1.0) for i in range(30)]
    w = vf.windows(30)
    for c, s, m in zip(w["centres"], w["samples"], w["markers"]):
        assert refs[s[m]] == refs[c]


# ----------------------------------------------------------------------------------------------- float32 batches
@dataclass
class _VF:
    n: int
    feat: np.ndarray
    prob: np.ndarray


class _Store:
    def __init__(self, lengths: dict[str, int]) -> None:
        self.lengths = lengths

    def get(self, vid: str, variant: str) -> _VF:
        n = self.lengths[vid]
        return _VF(n, np.zeros((n, 3), np.float32), np.zeros((n, 5), np.float32))


def _toy_training():
    rng = np.random.default_rng(3)
    lengths = {f"c_{k}": int(rng.integers(9, 400)) for k in range(30)}
    clips = [(v, int(rng.integers(0, 5))) for v in lengths]
    streams = []
    for k in range(10):
        n = int(rng.integers(48, 120)); lab = rng.integers(0, 5, n).astype(np.int64)
        streams.append(vf.Seq(f"multi_train_{k:04d}", False, rng.normal(size=(n, 3)).astype(np.float32), lab, np.array([n // 2]), None))
    native = {v: rng.normal(size=(n, 3)).astype(np.float32) for v, n in lengths.items()}
    return lengths, clips, streams, native


def test_the_float32_adapter_draws_the_same_batches_as_train_data():
    lengths, clips, streams, native = _toy_training()
    old = TrainData(tuple(streams), tuple(clips), _Store(lengths))
    new = vf.Batches(streams, [(v, lab, lengths[v]) for v, lab in clips], native)
    for seed in (0, 2):
        for epoch in (0, 1, 7):
            a = old.epoch_batches(seed, epoch, 12, 256, 4, 4); b = new.epoch(seed, epoch, 12, 256, 4, 4)
            assert [[s.seq_id for s in x] for x in a] == [[s.seq_id for s in x] for x in b]
            assert [[s.labels.size for s in x] for x in a] == [[s.labels.size for s in x] for x in b]
            assert all(np.array_equal(p.labels, q.labels) and np.array_equal(p.joins, q.joins) for x, y in zip(a, b) for p, q in zip(x, y))


def test_native_crops_are_float32_slices_of_the_whole_recording():
    lengths, clips, streams, native = _toy_training()
    b = vf.Batches(streams, [(v, lab, lengths[v]) for v, lab in clips], native).epoch(1, 0, 12, 256, 4, 4)
    crops = [s for x in b for s in x if s.is_native]
    assert len(crops) == 12
    for s in crops:
        vid, start = s.seq_id.rsplit("@", 1); start = int(start)
        assert s.feat.dtype == np.float32 and np.array_equal(s.feat, native[vid][start:start + s.labels.size])
        assert s.labels.size == min(256, lengths[vid]) and s.joins.size == 0
    x = collate(b[0])["x"].numpy()
    assert x.dtype == np.float32 and np.array_equal(x[0, :, :b[0][0].labels.size], b[0][0].feat.T)   # no fp16 rounding on the way


# ----------------------------------------------------------------------------------------------- training-only index
def test_the_index_read_returns_training_rows_and_never_parses_other_labels(tmp_path):
    p = tmp_path / "index.csv"
    p.write_text("video_id,split,raw_label,label_index,n_samples,path\n"
                 "t_1,train,A4C,2,40,x\n"
                 "v_1,validation,SECRET,not-a-number,50,y\n"
                 "t_2,train,A5C,3,41,z\n"
                 "v_2,validation,SECRET,also-not,52,w\n")
    df = vf.training_index(p)
    assert df["video_id"].tolist() == ["t_1", "t_2"] and df["label_index"].tolist() == [2, 3]
    assert df["n_samples"].tolist() == [40, 41] and "raw_label" not in df.columns and "path" not in df.columns
    assert set(df["split"]) == {"train"}
