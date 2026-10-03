"""Round-3 boundary-only branch: decoding with the frozen classifier, loss, padding invariance, no class path."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from echo_routing.temporal.dev_data import Seq, collate
from echo_routing.temporal.dev_heads import BoundaryOnlyHead
from echo_routing.temporal.segments import boundary_decode

D = 12


def _onehot(labels, p=0.9):
    out = np.full((len(labels), 5), (1 - p) / 4); out[np.arange(len(labels)), labels] = p
    return out


def test_boundary_decode_uses_peaks_and_the_classifier_mean():
    prob = _onehot([0] * 6 + [3] * 2 + [0] * 2 + [2] * 10)     # a 2-sample flicker inside the first interval
    b = np.zeros(20); b[10] = 0.95; b[11] = 0.6; b[0] = 1.0     # one confident peak at 10, a weaker neighbour, position 0 ignored
    lab = boundary_decode(prob, b, threshold=0.5, nms_radius=5)
    assert lab.tolist() == [0] * 10 + [2] * 10                   # flicker removed, change kept, NMS drops 11


def test_adjacent_intervals_with_the_same_label_merge():
    prob = _onehot([1] * 20); b = np.zeros(20); b[10] = 0.9
    assert boundary_decode(prob, b, 0.5, 5).tolist() == [1] * 20


def _seq(n, seed, labels, joins=(), native=False, sid="s"):
    rng = np.random.default_rng(seed)
    return Seq(sid, native, rng.normal(size=(n, D)).astype(np.float16), np.asarray(labels), np.asarray(joins, int), _onehot(labels))


def test_head_trains_on_boundaries_only_and_labels_come_from_the_classifier():
    head = BoundaryOnlyHead({"in_dim": D, "dim": 16, "dilations": [1, 2, 4], "dropout": 0.0}, threshold=0.5, nms_radius=5)
    torch.manual_seed(0); model = head.build()
    seqs = [_seq(12, 1, [0] * 6 + [2] * 6, [6], sid="a"), _seq(7, 2, [4] * 7, native=True, sid="b")]
    loss, parts = head.loss(model, collate(seqs)); assert torch.isfinite(loss) and set(parts) == {"boundary"}
    loss.backward()
    out = head.decode(model, seqs, "cpu")
    for s in seqs:   # every label is a classifier class taken over an interval, never a network output
        assert set(out[s.seq_id]["labels"].tolist()) <= set(s.labels.tolist()) and np.array_equal(out[s.seq_id]["prob"], s.prob)
    assert not any("cls" in n or "gate" in n or "attn" in n for n, _ in model.named_parameters())


def test_boundary_only_head_requires_every_size_in_the_config():
    with pytest.raises(KeyError):
        BoundaryOnlyHead({"in_dim": D, "dim": 16}, 0.5, 5).build()


def test_boundary_only_padding_invariance():
    head = BoundaryOnlyHead({"in_dim": D, "dim": 16, "dilations": [1, 2, 4], "dropout": 0.2}, 0.5, 5); torch.manual_seed(1); model = head.build().eval()
    short, long_ = _seq(7, 3, [1] * 7, sid="x"), _seq(25, 4, [2] * 25, sid="y")
    a = model(*(lambda b: (b["x"], b["mask"]))(collate([short])))["boundary_logits"][0, 0, :7]
    b = model(*(lambda c: (c["x"], c["mask"]))(collate([short, long_])))["boundary_logits"][0, 0, :7]
    assert torch.allclose(a, b, atol=1e-5)
