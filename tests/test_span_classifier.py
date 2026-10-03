"""Round-6 span classifier: span records, frame caps, decisions, the head, batches and the schedule."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from echo_routing.temporal.span_classifier import (A4C, A5C, SpanHead, apply_decisions, epoch_order, even_cap, lr_factor,
                                                   span_means, span_records, target_index)
from echo_routing.temporal.windowed import SampleRef


def _refs(n, vids):
    return [SampleRef(vids[i], i, 1.0) for i in range(n)]


def test_even_cap():
    assert even_cap(5, 32).tolist() == list(range(5))
    c = even_cap(100, 32); assert len(c) == 32 and c[0] == 0 and c[-1] == 99 and (np.diff(c) > 0).all()


def test_span_records_keep_only_a4c_a5c_votes_with_majority_purity_and_source():
    final = np.array([0] * 4 + [2] * 6 + [3] * 5 + [2] * 5)
    ref = np.array([0] * 4 + [2] * 4 + [3] * 2 + [3] * 5 + [1] * 5)
    vids = ["p"] * 4 + ["a"] * 4 + ["b"] * 7 + ["c"] * 5
    recs = span_records("s", final, [4, 10, 15], ref, _refs(20, vids), cap=3)
    assert [(r.start, r.end, r.vote, r.label) for r in recs] == [(4, 10, 2, 2), (10, 15, 3, 3), (15, 20, 2, -1)]
    assert recs[0].purity == pytest.approx(4 / 6) and recs[0].source == "a" and len(recs[0].refs) == 3
    with pytest.raises(ValueError):
        span_records("s", final, [4, 12], ref, _refs(20, vids), cap=3)   # the vote varies inside 4-12


def test_apply_decisions_changes_only_the_records():
    final = np.array([0] * 4 + [2] * 6 + [3] * 5)
    recs = span_records("s", final, [4, 10], final, _refs(15, ["v"] * 15), cap=32)
    out = apply_decisions(final, recs, [A5C, A4C])
    assert out.tolist() == [0] * 4 + [3] * 6 + [2] * 5
    with pytest.raises(ValueError):
        apply_decisions(final, recs, [0, A4C])


def test_head_means_order_schedule_and_targets():
    torch.manual_seed(0); h1 = SpanHead(); torch.manual_seed(0); h2 = SpanHead()
    assert all(torch.equal(a, b) for a, b in zip(h1.state_dict().values(), h2.state_dict().values()))
    x = torch.arange(10, dtype=torch.float32).view(5, 2)
    assert span_means(x, [2, 3]).tolist() == [[1.0, 2.0], [6.0, 7.0]]
    assert (epoch_order(50, 1, 2) == epoch_order(50, 1, 2)).all() and not (epoch_order(50, 1, 2) == epoch_order(50, 1, 3)).all()
    assert lr_factor(0, 10, 100) == pytest.approx(0.1) and lr_factor(10, 10, 100) == pytest.approx(1.0) and lr_factor(100, 10, 100) == pytest.approx(0.0)
    assert target_index(np.array([A4C, A5C])).tolist() == [0, 1]
    with pytest.raises(ValueError):
        target_index(np.array([0]))
